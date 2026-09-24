'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const video = $('exampleVideo');
  const tabs = [...document.querySelectorAll('[role="tab"][data-stage]')];
  const layerButtons = [...document.querySelectorAll('[data-layer]')];
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  let trace = null, stage = 0, layer = 0, time = 0, playing = false, lastTick = 0, raf = 0;
  let tour = false, tourStarted = 0, speech = null;
  const stages = [
    {title:'Start with movement.', text:'One complete utterance keeps the hands, face and body in context. Translation starts after recording ends.', metric:'25', unit:'fps', label:'The model’s training time grid', note:'Auslan has its own grammar. It is not English expressed through gestures.', next:'Next: extract the pose', visual:'CAPTURE / ONE COMPLETE UTTERANCE'},
    {title:'Keep the visual signal.', text:'DWPose locates the body, face and hands in every frame. Their coordinates form a sequence of movement.', metric:'133', unit:'points', label:'Body · feet · face · both hands', note:'These are visual coordinates, not recognised signs. Weak detections can lose important detail.', next:'Next: follow the features', visual:'EXTRACT / BODY, FACE & HANDS'},
    {title:'Build temporal context.', text:'A learned projection and four Transformer encoder layers combine information across the complete utterance.', metric:'4', unit:'layers', label:'Transformer encoder · 256 channels', note:'Feature values are not a map of meaning. No sign-to-word alignment is claimed.', next:'Next: decode the English', visual:'ENCODE / A SEQUENCE OF FEATURES'},
    {title:'Decode a sentence.', text:'The decoder searches possible English sequences, conditioned on the encoded movement. It never needs an intermediate gloss sequence.', metric:'10', unit:'beams', label:'Pose-only model · recorded operating point', note:'This walkthrough follows the pose-only model. S6 separately combines three pose-only and three image-fusion translators.', next:'Next: decide when to speak', visual:'DECODE / ENGLISH AFTER THE UTTERANCE'},
    {title:'Review. Then speak.', text:'A confidence threshold decides whether to offer an answer. The person reviews the English and chooses whether to hear it.', metric:'You', unit:'decide', label:'Speech starts only on request', note:'Confidence is not a probability of correctness. A fluent spoken sentence can still be wrong.', next:'Back to the beginning', visual:'RESPOND / OPTIONAL SPEECH'}
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
    if (!speech) text('speechStatus', !trace ? 'Speech becomes available with a recorded example.' : !trace.accepted ? 'This result does not clear the response threshold; speech is withheld.' : voices.length ? 'Uses a local English voice on this device. No automatic playback.' : 'No local English voice is available in this browser. The text remains readable.');
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
    text('visualContext', trace ? (stage < 2 ? 'Video + stored DWPose · same frame' : stage === 2 ? 'Measured features · not attention' : 'Recorded decode · not live inference') : 'Schematic · no inference result');
    if (trace && stage === 3) setMetric(String(trace.beam), 'beams');
    if (!trace && stage === 3) text('stageMetricLabel', 'Single-model decoding configuration');
    if (stage === 2) {
      document.querySelector('.feature-key').hidden = !trace;
      document.querySelector('.matrix-axis').hidden = !trace;
      document.querySelector('.matrix-y').textContent = trace ? 'FEATURE CHANNELS' : 'ARCHITECTURE';
      setLayer(layer);
    }
    paint();
  }
  function setLayer(next) {
    layer = next;
    layerButtons.forEach((b, i) => b.setAttribute('aria-pressed', String(i === layer)));
    if (trace) {
      text('featureNote', trace.featureLabels[layer] + ' · ' + trace.frames + ' time steps × ' + trace.features[layer].width + ' channels. Colours show 24 channel-group means, scaled per panel.');
      $('featureCanvas').setAttribute('aria-label', trace.featureLabels[layer] + ' measured feature matrix. ' + trace.frames + ' time steps, ' + trace.features[layer].width + ' channels, displayed as 24 channel-group means.');
      setMetric(String(trace.features[layer].width), 'channels');
      text('stageMetricLabel', trace.featureLabels[layer]);
    }
    drawFeatures();
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
  function drawFeatures(){
    if(stage!==2)return;
    const s=surface('featureCanvas');if(!s)return;
    const {ctx,w,h}=s;
    if(!trace){
      // Public schematic: architecture blocks, never made-up activation values.
      const names=['399-D input','256-D projection','Encoder 1','Encoder 2','Encoder 3','Encoder 4'];
      const gap=8,bw=(w-gap*5)/6;
      names.forEach((name,i)=>{
        const bh=(i? .7:.95)*h,y=(h-bh)/2;
        ctx.fillStyle=i===layer?'#b8d797':'#35544c';ctx.fillRect(i*(bw+gap),y,bw,bh);
        ctx.strokeStyle=i===layer?'#e1f2c1':'#6f928071';ctx.strokeRect(i*(bw+gap)+.5,y+.5,bw-1,bh-1);
        ctx.save();ctx.translate(i*(bw+gap)+bw/2,h/2);ctx.rotate(-Math.PI/2);
        ctx.fillStyle=i===layer?'#183c35':'#c8d9c8';ctx.font='10px ui-monospace,monospace';ctx.textAlign='center';ctx.fillText(name,0,3);ctx.restore();
      });
      return;
    }
    const values=trace.features[layer].values, cw=w/values.length,ch=h/24;
    values.forEach((column,x)=>column.forEach((v,y)=>{
      const base=[29,55,48],target=v<0?[128,151,211]:[217,238,159],a=Math.min(1,Math.abs(v));
      ctx.fillStyle='rgb('+base.map((c,i)=>Math.round(c+(target[i]-c)*a)).join(',')+')';
      ctx.fillRect(x*cw,y*ch,Math.max(1,cw-.6),Math.max(1,ch-.6));
    }));
    ctx.strokeStyle='#ffffff';ctx.lineWidth=1.3;ctx.strokeRect(frame()*cw+.5,.5,Math.max(1,cw-1),h-1);
  }
  function paint(){
    const d=duration();
    $('timeline').value=String(Math.round(time/d*1000));
    $('timeline').setAttribute('aria-valuetext',time.toFixed(2)+' of '+d.toFixed(2)+' seconds'+(trace?', frame '+(frame()+1):' of illustration'));
    text('timeLabel',time.toFixed(2)+' / '+d.toFixed(2)+' s');
    text('frameLabel',stage>2?'UTTERANCE COMPLETE':trace?'FRAME '+String(frame()+1).padStart(3,'0')+' / '+trace.frames:'SCHEMATIC');
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
  $('nextStage').addEventListener('click',()=>setStage((stage+1)%5));
  $('playButton').addEventListener('click',()=>{endTour();if(playing)pause();else startPlayback();});
  $('timeline').addEventListener('input',()=>{stop();time=Number($('timeline').value)/1000*duration();if(trace)video.currentTime=time;paint();});
  video.addEventListener('seeked',()=>{if(!playing){time=Math.min(video.currentTime,duration());paint();}});
  video.addEventListener('ended',()=>{playing=false;time=duration();text('playIcon','▶');$('playButton').setAttribute('aria-label','Replay recorded clip');paint();});
  $('tourButton').addEventListener('click',()=>{
    if(reducedMotion.matches){setStage((stage+1)%5);return;}
    if(tour){stop();return;}
    stop();setStage(0);time=0;video.currentTime=0;tour=true;tourStarted=performance.now();
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
    if(data.kind!=='recorded'||data.version!==1||!Number.isInteger(data.frames)||data.frames<2||data.frames>300
       ||!finite(data.fps)||data.fps<=0||data.fps>60||!finite(data.duration)||data.duration<=0||data.duration>15
       ||!finite(data.width)||!finite(data.height)||data.width<=0||data.height<=0
       ||!finite(data.confidence)||!finite(data.threshold)||typeof data.accepted!=='boolean'
       ||typeof data.prediction!=='string'||typeof data.reference!=='string'
       ||data.video!=='clip.mp4'||data.poster!=='poster.jpg')throw new Error('Invalid example metadata.');
    if(!Array.isArray(data.points)||data.points.length!==data.frames
       ||data.points.some(row=>!Array.isArray(row)||row.length!==133||row.some(p=>!Array.isArray(p)||p.length!==3||p.some(v=>!finite(v)))))throw new Error('Invalid pose sequence.');
    if(!Array.isArray(data.features)||data.features.length!==6||!Array.isArray(data.featureLabels)||data.featureLabels.length!==6
       ||data.features.some(m=>!Array.isArray(m.values)||m.values.length!==data.frames||m.values.some(row=>row.length!==24||row.some(v=>!finite(v)))))throw new Error('Invalid feature sequence.');
    if(Math.abs(data.frames/data.fps-data.duration)>.05||data.accepted!==(data.confidence<=data.threshold))throw new Error('Inconsistent example metadata.');
    return data;
  }
  async function loadExample(){
    // Dataset media is only served by the explicit loopback-only preview helper.
    const local=['127.0.0.1','localhost','[::1]'].includes(location.hostname);
    if(!local||new URLSearchParams(location.search).get('example')!=='local')return;
    text('exampleBadge','Loading recorded example…');
    try{
      const response=await fetch('/research-example/manifest.json',{signal:AbortSignal.timeout(10000)});
      if(!response.ok)throw new Error('Example unavailable');
      trace=validateTrace(await response.json());
      video.src='/research-example/'+trace.video;video.poster='/research-example/'+trace.poster;
      $('videoPane').hidden=false;
      text('exampleBadge',trace.duration.toFixed(2)+' s · recorded example');
      text('poseLabel','DWPose / '+trace.frames+' MATCHED FRAMES');
      $('poseCanvas').setAttribute('aria-label','Extracted DWPose body, face and hand points for the same frame as the signing video.');
      text('predictionLabel','MODEL PREDICTION');
      text('predictionText',trace.prediction);
      text('referenceText',trace.reference);
      $('referenceBlock').hidden=false;
      text('decodeMode','Saved inference result');
      text('decisionText',(trace.accepted?'Clears':'Does not clear')+' the response threshold · self-CE '+trace.confidence.toFixed(3)+' / '+trace.threshold.toFixed(3)+' nats. Lower is more confident, not necessarily correct.');
      text('exampleCaption','One selected validation clip, followed through a pose-only Transformer. Stored-pose replay; not webcam inference or representative performance.');
      $('provenanceText').replaceChildren();
      for(const line of [trace.provenance,trace.feature_note,'Clip: '+trace.clip+' · '+trace.split+'. Model: '+trace.model+'. Beam '+trace.beam+', length penalty '+trace.length_penalty+'.','Checkpoint SHA-256: '+trace.checkpoint_sha256]){
        const p=document.createElement('p');p.textContent=line;p.style.overflowWrap='anywhere';$('provenanceText').append(p);
      }
      text('playIcon','▶');setStage(stage);updateVoice();paint();
    }catch(e){
      trace=null;text('exampleBadge','Interactive schematic');text('exampleError','The local recorded example is unavailable. Showing the clearly labelled schematic instead. Run the local walkthrough exporter and preview server to inspect the recording.');$('exampleError').hidden=false;
    }
  }

  const values={official:[9.522,10.272,10.335,11.802],unseen:[4.88,5.31,5.81,6.60]};
  const labels=['DWPose plus CTC','Pose–text pretraining','Image fusion','Six-model ensemble'];
  document.querySelectorAll('input[name="subset"]').forEach(input=>input.addEventListener('change',()=>{
    document.querySelectorAll('.bar').forEach((bar,i)=>{bar.style.width=values[input.value][i]/12.5*100+'%';bar.querySelector('b').textContent=values[input.value][i].toFixed(2);});
    $('resultsChart').setAttribute('aria-label',(input.value==='unseen'?'Unseen English targets':'Official test')+' BLEU-4: '+labels.map((label,i)=>label+' '+values[input.value][i].toFixed(2)).join(', ')+'.');
    text('chartCaption',input.value==='unseen'?'570 clips with English references absent from training. This is not an unseen-signer evaluation. See the uncertainty notes below.':'Saved decodes on the official conversational test. S6 combines three pose-only and three image-fusion models.');
  }));
  // Current operating points come from the audited snapshot, not duplicated numbers.
  fetch('/static/evidence.json').then(r=>{if(!r.ok)throw new Error();return r.json();}).then(data=>{
    const points=data.live_system?.operating_points;if(!points)return;
    for(const point of Object.values(points)){
      const p=document.createElement('p');
      const score=point.official_test_bleu4 ?? point.test_bleu4;
      p.textContent=point.label+': '+(score!==undefined?score.toFixed(3)+' BLEU-4 on the official test. ':'')+(point.recipe||'')+' '+(point.selection||'');
      $('liveEvidence').append(p);
      const readout=point.test_readout_at_validation_threshold?.['0.5'];
      if(readout){const q=document.createElement('p');q.textContent='At its validation-fitted 50% answer-rate threshold, test coverage is '+(readout.answered*100).toFixed(1)+'% and BLEU-4 on answered clips is '+readout.answered_bleu4.toFixed(2)+'. This selected-subset score is not comparable to full-coverage results.';$('liveEvidence').append(q);}
    }
  }).catch(()=>{}); // The static evidence link and core results remain usable.
  endTour();setStage(0);updateVoice();loadExample();
})();
