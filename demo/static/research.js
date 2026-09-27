'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const video = $('exampleVideo');
  const tabs = [...document.querySelectorAll('[role="tab"][data-stage]')];
  const layerButtons = [...document.querySelectorAll('[data-layer]')];
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  let trace = null, stage = 0, layer = 0, time = 0, playing = false, lastTick = 0, raf = 0;
  let tour = false, tourStarted = 0, speech = null;
  let inspecting = false, channel = 0, point = 91, matrixCache = null;
  const stages = [
    {title:'Watch the whole sentence.', text:'Hands, face and body work together. The translator waits for the recording to finish before producing English.', metric:'25', unit:'frames / second', label:'Still images that make up the video', note:'Auslan has its own grammar and ways of expressing meaning.', next:'Next: track the movement', visual:'WATCH / THE COMPLETE RECORDING'},
    {title:'Track what moves.', text:'A tracker called DWPose marks points on the body, face and hands. These moving dots become the translator’s input.', metric:'133', unit:'points', label:'Tracked in each video frame', note:'Tracking shows where things move. It does not recognise individual signs.', next:'Next: find patterns', visual:'TRACK / BODY, FACE & HANDS'},
    {title:'Connect movement over time.', text:'The model combines information across video frames. Each layer builds on the patterns found by the previous one.', metric:'4', unit:'layers', label:'Four stages of processing', note:'The diagrams show internal calculations, not a meaning assigned to each sign.', next:'Next: write the English', visual:'CONNECT / PATTERNS ACROSS FRAMES'},
    {title:'Generate an English sentence.', text:'Using the complete movement sequence, the model builds an English translation. Compare its output with the reference below.', metric:'1', unit:'sentence', label:'Generated after the recording ends', note:'The reference is the dataset’s English translation. The model’s wording can be different or wrong.', next:'Next: review before speaking', visual:'TRANSLATE / MOVEMENT INTO WORDS'},
    {title:'Review. Then speak.', text:'When confidence is high enough, the translation is offered for review. You choose whether to hear it spoken aloud.', metric:'You', unit:'decide', label:'Speech starts only on request', note:'High confidence does not guarantee a correct translation.', next:'Back to the beginning', visual:'SPEAK / ONLY AFTER REVIEW'}
  ];
  const duration = () => trace ? trace.duration : 3;
  const frame = () => trace ? Math.min(trace.frames - 1, Math.floor(time * trace.fps + 1e-5)) : 0;
  const text = (id, value) => { $(id).textContent = value; };

  function setMetric(value, unit) {
    const el = $('stageMetric');
    el.replaceChildren(document.createTextNode(value + ' '));
    const small = document.createElement('small'); small.textContent = unit; el.append(small);
  }
  function stopSpeech() {
    if (speech && 'speechSynthesis' in window) window.speechSynthesis.cancel();
    speech = null;
    $('exhibit').classList.remove('speaking');
    $('stopSpeech').hidden = true;
    updateVoice();
  }
  function updateVoice() {
    const voices = 'speechSynthesis' in window ? window.speechSynthesis.getVoices().filter(v => v.localService && /^en\b/i.test(v.lang)) : [];
    $('speakButton').disabled = !trace || !trace.accepted || !voices.length || Boolean(speech);
    if (!speech) text('speechStatus', !trace ? 'Speech becomes available with a recorded example.' : !trace.accepted ? 'Confidence is too low for speech in this example.' : voices.length ? 'Uses an English voice on your device. Plays only when you choose.' : 'No local English voice is available in this browser. The text remains readable.');
    return voices.find(v => v.lang.toLowerCase() === 'en-au') || voices[0];
  }
  function endTour() {
    tour = false;
    text('tourButton', reducedMotion.matches ? 'Step through the pipeline →' : 'Play walkthrough ↗');
  }
  function pause() {
    playing = false; video.pause();
    text('playIcon', '▶');
    $('playButton').setAttribute('aria-label', trace ? 'Play recorded clip' : 'Play illustration');
    cancelAnimationFrame(raf); raf = 0;
  }
  function stop() { endTour(); pause(); }
  function setStage(next, manual = true) {
    if (manual) stop();
    stopSpeech();
    stage = next;
    $('exhibit').dataset.stage = String(stage);
    tabs.forEach((tab, i) => { tab.setAttribute('aria-selected', String(i === stage)); tab.tabIndex = i === stage ? 0 : -1; });
    $('stage-panel').setAttribute('aria-labelledby', 'tab-' + stage);
    const s = stages[stage];
    text('stageCounter', String(stage + 1).padStart(2, '0') + ' / 05');
    text('stageTitle', s.title); text('stageDescription', s.text);
    setMetric(s.metric, s.unit); text('stageMetricLabel', s.label); text('stageFootnote', s.note);
    text('nextStage', s.next + ' →'); text('visualLabel', s.visual);
    $('captureView').hidden = stage > 1;
    $('featureView').hidden = stage !== 2;
    $('languageView').hidden = stage < 3;
    $('speechControls').hidden = stage !== 4;
    $('poseLegend').hidden = stage > 1;
    $('timeline').disabled = stage > 2;
    $('playButton').disabled = stage > 2;
    text('visualContext', trace ? (stage < 2 ? 'Same moment · video and tracking' : stage === 2 ? 'Recorded model calculations' : 'Saved model output') : 'Illustration · no recorded result');
    if (!trace && stage === 3) text('stageDescription','Using the complete movement sequence, the model builds an English translation. A recorded example lets you compare it with a reference.');
    if (stage === 2) setLayer(layer);
    paint();
  }
  const layerStories = [
    ['01 / REPRESENT', 'Turn a frame into numbers.', 'Each dot contributes its horizontal position, vertical position and tracking confidence.', '133 × 3 → 399'],
    ['02 / PROJECT', 'Find useful combinations.', 'A learned calculation turns the 399 input numbers into 256 features for each frame.', '399 → 256'],
    ['03 / CONNECT', 'Look across the recording.', 'Attention lets each frame draw information from other moments in the video.', '256 → 256'],
    ['04 / REFINE', 'Look again, with context.', 'The second layer combines frame information already processed by the first.', '256 → 256'],
    ['05 / REFINE', 'Build on earlier patterns.', 'The third layer updates each frame using information from across the recording.', '256 → 256'],
    ['06 / ENCODE', 'Prepare to generate words.', 'After the fourth layer, the processed sequence is passed to the English sentence generator.', '256 → 256']
  ];
  function setFeatureMode(value) {
    inspecting = Boolean(value && trace);
    $('featureExplain').setAttribute('aria-pressed', String(!inspecting));
    $('featureInspect').setAttribute('aria-pressed', String(inspecting));
    $('featureInspect').disabled = !trace;
    $('featureExplanation').hidden = inspecting;
    $('featureInspection').hidden = !inspecting;
    text('featureModeLabel', inspecting ? 'ALL FRAMES · ALL CHANNELS' : 'FOLLOW ONE FRAME');
    drawFeatures();
  }
  function setLayer(next) {
    layer = next;
    layerButtons.forEach((b, i) => b.setAttribute('aria-pressed', String(i === layer)));
    const story = layerStories[layer];
    text('featureOperationLabel',story[0]); text('featureHeading',story[1]); text('featureSummary',story[2]); text('featureEquation',story[3]);
    $('coordinatePanel').hidden = layer !== 0;
    $('featureDiagram').hidden = layer === 0;
    $('featurePoint').disabled = !trace;
    const width = layer ? 256 : 399;
    channel = Math.min(channel, width - 1);
    $('featureChannel').max = String(width - 1);
    $('featureChannel').value = String(channel);
    setMetric(layer ? '256' : '399', 'values / frame');
    text('stageMetricLabel', trace ? trace.featureLabels[layer] : layer === 0 ? '133 points × (x, y, confidence)' : 'One vector per time step');
    if (trace) {
      $('featureCanvas').setAttribute('aria-label', trace.featureLabels[layer] + ' measured feature matrix. ' + trace.frames + ' time steps, ' + width + ' individual channels. Use the timeline and channel slider to inspect exact values.');
      text('matrixEnd', trace.duration.toFixed(2) + ' s');
      const scale = trace.features[layer].scale;
      text('featureMin', '−' + scale.toFixed(2)); text('featureMax', '+' + scale.toFixed(2));
    }
    setFeatureMode(inspecting);
  }

  // COCO-WholeBody-133 topology. Low-confidence points remain hidden.
  const edges = [[0,1],[0,2],[1,3],[2,4],[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16],[15,17],[15,18],[15,19],[16,20],[16,21],[16,22]];
  const chain = (a,b,closed=false) => { for(let i=a;i<b;i++) edges.push([i,i+1]); if(closed) edges.push([b,a]); };
  [[23,39],[40,44],[45,49],[50,53],[54,58]].forEach(([a,b]) => chain(a,b));
  [[59,64],[65,70],[71,82],[83,90]].forEach(([a,b]) => chain(a,b,true));
  for (const root of [91,112]) for(let finger=0;finger<5;finger++) {
    let prev=root; for(let j=1;j<=4;j++){ const current=root+finger*4+j;edges.push([prev,current]);prev=current; }
  }
  edges.push([9,91],[10,112]);
  function surface(id) {
    const canvas = $(id), rect = canvas.getBoundingClientRect(), dpr = Math.min(window.devicePixelRatio || 1,2);
    const w = rect.width, h = rect.height;
    if (!w || !h) return null;
    if (canvas.width !== Math.round(w*dpr) || canvas.height !== Math.round(h*dpr)) {
      canvas.width = Math.round(w*dpr); canvas.height = Math.round(h*dpr);
    }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,w,h);
    return {ctx,w,h};
  }
  function drawPose(id, overlay = false) {
    const s=surface(id); if (!s) return;
    const {ctx,w,h}=s;
    if (!trace) { drawSchematic(ctx,w,h); return; }
    const scale=Math.min(w/trace.width,h/trace.height), ox=(w-trace.width*scale)/2, oy=(h-trace.height*scale)/2;
    const points=trace.points[frame()], xy=i=>[points[i][0]*scale+ox,points[i][1]*scale+oy];
    ctx.lineWidth = overlay ? 1.5 : 1.15;
    for(const [a,b] of edges){
      if(points[a][2]<.3 || points[b][2]<.3)continue;
      const p=xy(a),q=xy(b);
      ctx.strokeStyle=a>=91?'#d4ed96':a>=23?'#d9d3a495':'#8bc7b9';
      ctx.beginPath();ctx.moveTo(...p);ctx.lineTo(...q);ctx.stroke();
    }
    points.forEach((p,i)=>{
      if(p[2]<.3)return;
      ctx.fillStyle=i>=91?'#e0f6af':i>=23?'#e5d9a8':'#a5ddcd';
      ctx.beginPath();ctx.arc(...xy(i),i>=23?1.3:2.7,0,Math.PI*2);ctx.fill();
    });
  }
  function drawSchematic(ctx,w,h) {
    // Designed illustration, deliberately not a fabricated signing example.
    const scale=Math.min(w/600,h/370),ox=w/2,oy=h*.51;
    ctx.save();ctx.translate(ox,oy);ctx.scale(scale,scale);
    const phase=time/3*Math.PI*2, lift=Math.sin(phase)*12;
    const body=[[-81,-22],[81,-22],[-121,62],[123,67],[-68,31+lift],[68,18-lift],[-65,154],[65,154]];
    ctx.strokeStyle='#a3c4b4';ctx.lineWidth=2;
    for(const [a,b] of [[0,1],[0,2],[2,4],[1,3],[3,5],[0,6],[1,7],[6,7]]){ctx.beginPath();ctx.moveTo(...body[a]);ctx.lineTo(...body[b]);ctx.stroke();}
    ctx.strokeStyle='#d7d3a0';ctx.lineWidth=1;
    for(let i=0;i<23;i++){const a=i/23*Math.PI*2;ctx.beginPath();ctx.arc(Math.cos(a)*35,-84+Math.sin(a)*44,1.5,0,Math.PI*2);ctx.stroke();}
    for(const [x,y] of [[-12,-88],[12,-88],[0,-75],[-9,-64],[0,-61],[9,-64]]){ctx.beginPath();ctx.arc(x,y,1.7,0,Math.PI*2);ctx.stroke();}
    body.forEach(p=>{ctx.fillStyle='#afd5c0';ctx.beginPath();ctx.arc(...p,4,0,Math.PI*2);ctx.fill();});
    for(let side=0;side<2;side++){
      const [x,y]=body[side+4];ctx.save();ctx.translate(x,y);ctx.rotate((side?1:-1)*.26);
      for(let f=0;f<5;f++){
        let px=0,py=0;
        for(let j=1;j<=4;j++){
          const nx=(f-2)*9*j/4,ny=-j*(f===0?8:12)+Math.abs(f-2)*3;
          ctx.strokeStyle='#d3efa2';ctx.lineWidth=1.4;ctx.beginPath();ctx.moveTo(px,py);ctx.lineTo(nx,ny);ctx.stroke();
          ctx.fillStyle='#d3efa2';ctx.beginPath();ctx.arc(nx,ny,2,0,Math.PI*2);ctx.fill();px=nx;py=ny;
        }
      }ctx.restore();
    }
    ctx.strokeStyle='#8cb49c45';ctx.lineWidth=1;
    for(const [x,y,dx,dy] of [[-180,-144,1,1],[180,-144,-1,1],[-180,157,1,-1],[180,157,-1,-1]]){ctx.beginPath();ctx.moveTo(x+18*dx,y);ctx.lineTo(x,y);ctx.lineTo(x,y+18*dy);ctx.stroke();}
    ctx.font='10px ui-monospace,monospace';ctx.fillStyle='#adc3b2';ctx.fillText('FACE',73,-101);ctx.fillText('HANDS',-176,17);ctx.fillText('BODY',106,133);
    ctx.restore();
  }
  function featureColour(value, scale) {
    const base=[29,55,48], target=value<0?[146,155,229]:[217,238,159];
    const a=Math.min(1,Math.abs(value)/Math.max(scale,1e-6));
    return 'rgb('+base.map((c,i)=>Math.round(c+(target[i]-c)*a)).join(',')+')';
  }
  function drawFeatureFrame(){
    const s=surface('featureFrame'); if(!s)return;
    const {ctx,w,h}=s;
    if(!trace){drawSchematic(ctx,w,h);text('featureFrameLabel','ILLUSTRATION');$('featureFrame').setAttribute('aria-label','Illustrated pose, not a recorded signing frame.');return;}
    const scale=Math.min(w/trace.width,h/trace.height), ox=(w-trace.width*scale)/2, oy=(h-trace.height*scale)/2;
    // Never draw a stale video frame while a new seek is still pending.
    if(frame()===0 && $('featurePoster').complete && $('featurePoster').naturalWidth)ctx.drawImage($('featurePoster'),ox,oy,trace.width*scale,trace.height*scale);
    else if(frame()>0 && video.readyState>=2 && !video.seeking)ctx.drawImage(video,ox,oy,trace.width*scale,trace.height*scale);
    else drawPose('featureFrame');
    if(layer===0){
      const p=trace.points[frame()][point];
      ctx.strokeStyle=p[2]>=.3?'#e3ff8f':'#f2b491';ctx.lineWidth=2;
      ctx.beginPath();ctx.arc(p[0]*scale+ox,p[1]*scale+oy,5,0,Math.PI*2);ctx.stroke();
    }
    text('featureFrameLabel','FRAME '+String(frame()+1).padStart(3,'0')+' · '+(frame()/trace.fps).toFixed(2)+' s');
    $('featureFrame').setAttribute('aria-label','Source video frame '+(frame()+1)+(layer===0?', selected point highlighted.':'.'));
  }
  function drawFeatureFlow(){
    const s=surface('featureFlowCanvas');if(!s)return;
    const {ctx,w,h}=s;
    if(!trace){
      const names=layer===1?['Linear map','Normalise','ReLU']:['Attention','Add + norm','Feed-forward','Add + norm'];
      text('featureDiagramLabel',layer===1?'PER-FRAME PROJECTION':'INSIDE EACH ENCODER LAYER');
      text('featureDiagramScale','ARCHITECTURE');
      const bw=(w-(names.length-1)*15)/names.length;
      names.forEach((name,i)=>{
        const x=i*(bw+15);
        ctx.fillStyle='#365a4a';ctx.fillRect(x,32,bw,65);ctx.strokeStyle='#a5c981';ctx.strokeRect(x+.5,32.5,bw-1,64);
        ctx.fillStyle='#e3efca';ctx.font=(bw<70?'8':'9')+'px ui-monospace,monospace';ctx.textAlign='center';ctx.fillText(name,x+bw/2,69);
        if(i<names.length-1){ctx.fillStyle='#a5c981';ctx.fillText('→',x+bw+7.5,69);}
      });
      text('featureFlowCaption',layer===1?'Position information is added before the first encoder layer.':'Every frame can attend to the full utterance. Diagrams show operations, not measured weights.');
      $('featureFlowCanvas').setAttribute('aria-label',names.join(' → ')+'. Architecture only; no measured values.');
      return;
    }
    if(layer===1){
      const values=trace.features[1].values[frame()], scale=trace.features[1].scale;
      const cols=32,rows=8,cw=w/cols,ch=h/rows;
      values.forEach((v,i)=>{ctx.fillStyle=featureColour(v,scale);ctx.fillRect((i%cols)*cw,Math.floor(i/cols)*ch,Math.max(1,cw-1),ch-1);});
      text('featureDiagramLabel','256 LEARNED VALUES · THIS FRAME');text('featureDiagramScale','0 → '+scale.toFixed(2));
      text('featureFlowCaption','One cell per channel, ordered left to right. Brighter = larger. Position information is added next.');
      $('featureFlowCanvas').setAttribute('aria-label','All 256 projected feature values for frame '+(frame()+1)+'. Brighter cells represent larger values, not importance.');
      return;
    }
    const weights=trace.attention[layer-2][frame()], max=Math.max(...weights), peak=weights.indexOf(max), cw=w/weights.length;
    const qx=(frame()+.5)*cw, baseline=h-17, barHeight=56;
    // A fixed zero baseline, with the upper bound explicitly labelled per row.
    const top=weights.map((value,i)=>({value,i})).sort((a,b)=>b.value-a.value).slice(0,8);
    for(const {value,i} of top){
      const x=(i+.5)*cw;ctx.strokeStyle='rgba(210,238,161,'+(.15+.5*value/max)+')';ctx.lineWidth=.7+value/max;
      ctx.beginPath();ctx.moveTo(qx,15);ctx.bezierCurveTo(qx,45,x,32,x,baseline-value/max*barHeight);ctx.stroke();
    }
    weights.forEach((v,i)=>{ctx.fillStyle=i===frame()?'#e7f2d4':i===peak?'#e0f29d':'#82a982';ctx.fillRect(i*cw,baseline-v/max*barHeight,Math.max(1,cw-1),v/max*barHeight);});
    ctx.strokeStyle='#b4ceb3';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(0,baseline+.5);ctx.lineTo(w,baseline+.5);ctx.stroke();
    ctx.fillStyle='#e0f29d';ctx.beginPath();ctx.arc(qx,13,4,0,Math.PI*2);ctx.fill();
    ctx.font='8px ui-monospace,monospace';ctx.fillStyle='#bccfb6';ctx.textAlign='left';ctx.fillText('0 s',0,h-2);ctx.textAlign='right';ctx.fillText(trace.duration.toFixed(2)+' s',w,h-2);
    text('featureDiagramLabel','FRAME '+(frame()+1)+' → ALL '+trace.frames+' FRAMES');
    text('featureDiagramScale','0–'+(max*100).toFixed(1)+'%');
    text('featureFlowCaption','Mean of '+trace.attention_heads+' heads. Largest weight: frame '+(peak+1)+' ('+(max*100).toFixed(1)+'%). Lines show the eight largest weights.');
    $('featureFlowCanvas').setAttribute('aria-label','Encoder layer '+(layer-1)+' self-attention from frame '+(frame()+1)+' to every frame. Largest head-mean weight '+(max*100).toFixed(1)+' percent at frame '+(peak+1)+'. Not a sign-to-word alignment.');
  }
  function drawFeatures(){
    if(stage!==2)return;
    if(!inspecting){
      drawFeatureFrame();
      if(layer===0){
        const values=trace?trace.features[0].values[frame()].slice(point*3,point*3+3):null;
        ['coordinateX','coordinateY','coordinateConfidence'].forEach((id,i)=>text(id,values?values[i].toFixed(3):i===2?'0–1':'normalised'));
        text('coordinateNote',trace?'Original pixels: x '+trace.points[frame()][point][0].toFixed(1)+', y '+trace.points[frame()][point][1].toFixed(1)+'. Low-confidence coordinates (< 0.3) become zero.':'Coordinates are centred and scaled within each frame. Low-confidence coordinates become zero.');
      }else drawFeatureFlow();
      text('featureNote',layer===0?'Scrub the timeline: the highlighted point and its input values move together.':layer===1?'Learned channels have no assigned sign or word meaning.':trace?'Attention shows how this layer mixes frames, not correctness, causal importance or word alignment.':'Each layer keeps the sequence length and width; the representation changes.');
      return;
    }
    const s=surface('featureCanvas');if(!s||!trace)return;
    const {ctx,w,h}=s, matrix=trace.features[layer], values=matrix.values, cw=w/values.length,ch=h/matrix.width;
    const canvas=$('featureCanvas');
    if(!matrixCache || matrixCache.matrix!==matrix || matrixCache.canvas.width!==canvas.width || matrixCache.canvas.height!==canvas.height){
      const cached=document.createElement('canvas');cached.width=canvas.width;cached.height=canvas.height;
      const c=cached.getContext('2d'),dx=cached.width/values.length,dy=cached.height/matrix.width;
      values.forEach((column,x)=>column.forEach((v,y)=>{c.fillStyle=featureColour(v,matrix.scale);c.fillRect(x*dx,y*dy,Math.ceil(dx),Math.ceil(dy));}));
      matrixCache={matrix,canvas:cached};
    }
    ctx.drawImage(matrixCache.canvas,0,0,w,h);
    ctx.strokeStyle='#ffffff';ctx.lineWidth=1;ctx.strokeRect(frame()*cw+.5,.5,Math.max(1,cw-1),h-1);
    ctx.strokeStyle='#ffffff90';ctx.beginPath();ctx.moveTo(0,(channel+.5)*ch);ctx.lineTo(w,(channel+.5)*ch);ctx.stroke();
    text('featureChannelLabel',String(channel+1)); text('featureValue',values[frame()][channel].toFixed(5));
    $('featureChannel').setAttribute('aria-valuetext','Channel '+(channel+1)+', value '+values[frame()][channel].toFixed(5)+' at frame '+(frame()+1));
    text('featureNote',matrix.width+' individual channels × '+trace.frames+' frames. Click the matrix or use both sliders. Colour scale is specific to this layer; brightness is not importance.');
  }
  function paint(){
    const d=duration();
    $('timeline').value=String(Math.round(time/d*1000));
    $('timeline').setAttribute('aria-valuetext',time.toFixed(2)+' of '+d.toFixed(2)+' seconds'+(trace?', frame '+(frame()+1):' of illustration'));
    text('timeLabel',time.toFixed(2)+' / '+d.toFixed(2)+' s');
    text('frameLabel',stage>2?'RECORDING COMPLETE':trace?'FRAME '+String(frame()+1).padStart(3,'0')+' / '+trace.frames:'SCHEMATIC');
    if(stage<2){drawPose('poseCanvas');if(trace)drawPose('poseOverlay',true);}
    if(stage===2)drawFeatures();
  }
  function tick(now) {
    raf=0;
    if(!playing && !tour)return;
    const delta=Math.min((now-lastTick)/1000,.1);lastTick=now;
    if(playing){
      if(trace) time=Math.min(video.currentTime,duration());
      else time=Math.min(time+delta,duration());
      if(time>=duration()-.005){playing=false;video.pause();text('playIcon','▶');$('playButton').setAttribute('aria-label',trace?'Replay recorded clip':'Replay illustration');}
    }
    if(tour){
      const elapsed=(now-tourStarted)/1000,d=duration()+.6;
      const next=elapsed<d?0:elapsed<d*2?1:elapsed<d*2+6?2:elapsed<d*2+9?3:4;
      if(next!==stage){
        setStage(next,false);
        if(next<2){time=0;video.currentTime=0;startPlayback();}
        else {playing=false;video.pause();}
      }
      if(next===2)setLayer(Math.min(5,Math.floor(elapsed-d*2)));
      if(next===4){endTour();pause();}
    }
    paint();
    if((playing||tour)&&!raf)raf=requestAnimationFrame(tick);
  }
  function startPlayback(){
    if(stage>2)return;
    if(time>=duration()-.01){time=0;video.currentTime=0;}
    playing=true;lastTick=performance.now();text('playIcon','Ⅱ');
    $('playButton').setAttribute('aria-label',trace?'Pause recorded clip':'Pause illustration');
    if(trace)video.play().catch(()=>{stop();text('exampleError','The clip could not play in this browser. You can still scrub the recorded pose and inspect the saved features.');$('exampleError').hidden=false;});
    if(!raf)raf=requestAnimationFrame(tick);
  }
  tabs.forEach((tab,i)=>{
    tab.addEventListener('click',()=>setStage(i));
    tab.addEventListener('keydown',e=>{
      let next;
      if(e.key==='ArrowRight')next=(i+1)%tabs.length;
      if(e.key==='ArrowLeft')next=(i+tabs.length-1)%tabs.length;
      if(e.key==='Home')next=0;
      if(e.key==='End')next=tabs.length-1;
      if(next!==undefined){e.preventDefault();setStage(next);tabs[next].focus();}
    });
  });
  layerButtons.forEach((b,i)=>b.addEventListener('click',()=>{stop();setLayer(i);}));
  $('featureExplain').addEventListener('click',()=>setFeatureMode(false));
  $('featureInspect').addEventListener('click',()=>setFeatureMode(true));
  $('featurePoint').addEventListener('change',()=>{point=Number($('featurePoint').value);drawFeatures();});
  $('featureChannel').addEventListener('input',()=>{channel=Number($('featureChannel').value);drawFeatures();});
  $('featureCanvas').addEventListener('click',e=>{
    if(!trace || !inspecting)return;
    const r=$('featureCanvas').getBoundingClientRect();
    const selected=Math.max(0,Math.min(trace.frames-1,Math.floor((e.clientX-r.left)/r.width*trace.frames)));
    channel=Math.max(0,Math.min(trace.features[layer].width-1,Math.floor((e.clientY-r.top)/r.height*trace.features[layer].width)));
    $('featureChannel').value=String(channel);stop();time=selected/trace.fps;video.currentTime=time;paint();
  });
  $('nextStage').addEventListener('click',()=>setStage((stage+1)%5));
  $('playButton').addEventListener('click',()=>{endTour();if(playing)pause();else startPlayback();});
  $('timeline').addEventListener('input',()=>{stop();time=Number($('timeline').value)/1000*duration();if(trace)video.currentTime=time;paint();});
  $('featurePoster').addEventListener('load',paint);
  video.addEventListener('loadeddata',paint);
  video.addEventListener('seeked',()=>{if(!playing){time=Math.min(video.currentTime,duration());paint();}});
  video.addEventListener('ended',()=>{playing=false;time=duration();text('playIcon','▶');$('playButton').setAttribute('aria-label','Replay recorded clip');paint();});
  $('tourButton').addEventListener('click',()=>{
    if(reducedMotion.matches){setStage((stage+1)%5);return;}
    if(tour){stop();return;}
    stop();setFeatureMode(false);setStage(0);time=0;video.currentTime=0;tour=true;tourStarted=performance.now();
    text('tourButton','Pause walkthrough Ⅱ');startPlayback();
  });
  $('speakButton').addEventListener('click',()=>{
    const voice=updateVoice();if(!trace||!trace.accepted||!voice)return;
    stop();
    const utterance=new SpeechSynthesisUtterance(trace.prediction);utterance.voice=voice;utterance.lang=voice.lang;utterance.rate=1;
    speech=utterance;$('speakButton').disabled=true;$('stopSpeech').hidden=false;$('exhibit').classList.add('speaking');
    text('speechStatus','Reading with '+voice.name+' · '+voice.lang);
    utterance.onend=()=>{speech=null;$('stopSpeech').hidden=true;$('exhibit').classList.remove('speaking');updateVoice();};
    utterance.onerror=()=>{speech=null;$('stopSpeech').hidden=true;$('exhibit').classList.remove('speaking');updateVoice();text('speechStatus','Speech was interrupted. The English result remains available.');};
    window.speechSynthesis.speak(utterance);
  });
  $('stopSpeech').addEventListener('click',stopSpeech);
  if('speechSynthesis' in window)window.speechSynthesis.addEventListener('voiceschanged',updateVoice);
  document.addEventListener('visibilitychange',()=>{if(document.hidden){stop();stopSpeech();}});
  window.addEventListener('pagehide',()=>{stop();stopSpeech();});
  reducedMotion.addEventListener('change',()=>{stop();});
  if('IntersectionObserver' in window)new IntersectionObserver(entries=>{if(!entries[0].isIntersecting){stop();stopSpeech();}},{threshold:0}).observe($('exhibit'));
  if('ResizeObserver' in window)new ResizeObserver(paint).observe($('exhibit'));
  window.addEventListener('resize',paint);

  function validateTrace(data){
    const finite=v=>typeof v==='number'&&Number.isFinite(v);
    if(data.kind!=='recorded'||data.version!==2||!Number.isInteger(data.frames)||data.frames<2||data.frames>300
       ||!finite(data.fps)||data.fps<=0||data.fps>60||!finite(data.duration)||data.duration<=0||data.duration>15
       ||!finite(data.width)||!finite(data.height)||data.width<=0||data.height<=0
       ||!finite(data.confidence)||!finite(data.threshold)||typeof data.accepted!=='boolean'
       ||typeof data.prediction!=='string'||typeof data.reference!=='string'
       ||data.video!=='clip.mp4'||data.poster!=='poster.jpg')throw new Error('Invalid example metadata.');
    if(!Array.isArray(data.points)||data.points.length!==data.frames
       ||data.points.some(row=>!Array.isArray(row)||row.length!==133||row.some(p=>!Array.isArray(p)||p.length!==3||p.some(v=>!finite(v)))))throw new Error('Invalid pose sequence.');
    if(!Array.isArray(data.features)||data.features.length!==6||!Array.isArray(data.featureLabels)||data.featureLabels.length!==6
       ||data.features.some((m,i)=>m.width!==(i?256:399)||!finite(m.scale)||m.scale<=0||!Array.isArray(m.values)||m.values.length!==data.frames||m.values.some(row=>!Array.isArray(row)||row.length!==m.width||row.some(v=>!finite(v)))))throw new Error('Invalid feature sequence.');
    if(data.attention_heads!==8||!Array.isArray(data.attention)||data.attention.length!==4
       ||data.attention.some(matrix=>!Array.isArray(matrix)||matrix.length!==data.frames||matrix.some(row=>!Array.isArray(row)||row.length!==data.frames||row.some(v=>!finite(v)||v<0||v>1)||Math.abs(row.reduce((a,b)=>a+b,0)-1)>.0001)))throw new Error('Invalid attention sequence.');
    if(Math.abs(data.frames/data.fps-data.duration)>.05||data.accepted!==(data.confidence<=data.threshold))throw new Error('Inconsistent example metadata.');
    return data;
  }
  async function loadExample(){
    // The attributed, curated example is public. Arbitrary local exports are
    // still only read through the explicit loopback preview route.
    const mode=new URLSearchParams(location.search).get('example');
    if(mode==='schematic')return;
    const local=['127.0.0.1','localhost','[::1]'].includes(location.hostname);
    text('exampleBadge','Loading recorded example…');
    try{
      let base='/research-example/';
      if(!(local&&mode==='local')){
        const catalogResponse=await fetch('/static/examples/index.json',{signal:AbortSignal.timeout(10000)});
        if(!catalogResponse.ok)throw new Error('Example catalogue unavailable');
        const catalog=await catalogResponse.json();
        const examples=Array.isArray(catalog.examples)?catalog.examples.filter(item=>item && typeof item.id==='string' && /^[a-z0-9_]+$/.test(item.id)):[];
        if(!examples.length)throw new Error('Example catalogue is empty');
        const chosen=examples[Math.floor(Math.random()*examples.length)];
        base='/static/examples/'+chosen.id+'/';
      }
      const response=await fetch(base+'manifest.json',{signal:AbortSignal.timeout(10000)});
      if(!response.ok)throw new Error('Example unavailable');
      trace=validateTrace(await response.json());
      video.src=base+trace.video;video.poster=base+trace.poster;
      $('featurePoster').src=base+trace.poster;
      $('exampleCredit').hidden=false;
      $('videoPane').hidden=false;
      text('exampleBadge','Recorded example · '+trace.duration.toFixed(2)+' s');
      text('poseLabel','TRACKED MOVEMENT · DWPose');
      $('poseCanvas').setAttribute('aria-label','Extracted DWPose body, face and hand points for the same frame as the signing video.');
      text('predictionLabel','MODEL’S ENGLISH TRANSLATION');
      text('predictionText',trace.prediction);
      text('referenceText',trace.reference);
      $('referenceBlock').hidden=false;
      text('decodeMode','Recorded result');
      text('decisionText',trace.accepted?'Available for review. Confidence can still be wrong.':'Low model confidence: speech is withheld for this example.');
      text('exampleCaption','A real clip with recorded model results. This example illustrates the process, not typical accuracy.');
      text('exampleCreditClip',trace.clip+' · validation clip');
      $('provenanceText').replaceChildren();
      for(const line of [trace.provenance,trace.feature_note,'This walkthrough replays stored pose and model calculations; it does not run live inference. It follows one pose-only model, separate from the six-model benchmark in the Research Notes.','Clip: '+trace.clip+' · '+trace.split+'. Model: '+trace.model+'. Beam '+trace.beam+', length penalty '+trace.length_penalty+'.','Confidence: self-CE '+trace.confidence.toFixed(3)+' nats; response threshold '+trace.threshold.toFixed(3)+' nats. Lower is more confident, not necessarily correct.','Checkpoint SHA-256: '+trace.checkpoint_sha256]){
        const p=document.createElement('p');p.textContent=line;p.style.overflowWrap='anywhere';$('provenanceText').append(p);
      }
      text('playIcon','▶');setStage(stage);updateVoice();paint();
    }catch(e){
      trace=null;text('exampleBadge','Interactive schematic');text('exampleError','The recorded example could not be loaded. Showing a labelled schematic instead. Reload the page to try the video again.');$('exampleError').hidden=false;
    }
  }

  endTour();setStage(0);updateVoice();loadExample();
})();
