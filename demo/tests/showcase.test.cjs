'use strict';
// State-contract tests for the actual page controller. Browser visual/media QA
// is separate; these tests deliberately do not claim to decode video or audio.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.join(__dirname, '../static');
const source = fs.readFileSync(path.join(root, 'research.js'), 'utf8');
const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');

function fixture(accepted = true) {
  return {version:2,kind:'recorded',clip:'test-fixture',frames:74,fps:25,duration:2.96,width:778,height:838,
    video:'clip.mp4',poster:'poster.jpg',points:Array.from({length:74},()=>Array.from({length:133},()=>[1,2,.9])),
    features:Array.from({length:6},(_,i)=>({width:i?256:399,scale:1,values:Array.from({length:74},(_,f)=>Array.from({length:i?256:399},(_,c)=>f/100+c/1000))})),
    attention_heads:8,attention:Array.from({length:4},()=>Array.from({length:74},()=>Array(74).fill(1/74))),
    featureLabels:['Normalised input','Projection','Encoder 1','Encoder 2','Encoder 3','Encoder 4'],
    reference:'a fixture .',prediction:'a fixture.',confidence:accepted?.5:1.2,threshold:.7,accepted,
    beam:10,length_penalty:.7,model:'Test model',checkpoint_sha256:'test',provenance:'Test fixture',feature_note:'Test values'};
}
async function harness({hostname='127.0.0.1',search='',data=fixture(),reduced=false}={}) {
  const nodes=new Map(),pending=new Map(),requests=[],speechCalls=[];let now=0,nextID=1;
  const context2d=new Proxy({}, {get:()=>()=>{}});
  function element(id='') {
    return {id,hidden:false,disabled:false,textContent:'',value:'0',style:{},dataset:{},attributes:{},listeners:{},children:[],
      classList:{add(){},remove(){}},setAttribute(k,v){this.attributes[k]=v;},getAttribute(k){return this.attributes[k];},
      addEventListener(k,fn){this.listeners[k]=fn;},emit(k,e={}){return this.listeners[k]?.({...e,target:this});},
      append(child){this.children.push(child);},replaceChildren(...children){this.children=children;},
      getBoundingClientRect(){return {width:400,height:300,left:0,top:0};},getContext(){return context2d;},
      focus(){this.focused=true;},querySelector(){return this.valueNode ||= element();}};
  }
  for(const match of html.matchAll(/id="([^"]+)"/g))nodes.set(match[1],element(match[1]));
  for(const id of ['referenceBlock','exampleError','stopSpeech','exampleCredit'])nodes.get(id).hidden=true;
  const tabs=Array.from({length:5},(_,i)=>nodes.get('tab-'+i));
  const layers=Array.from({length:6},()=>element()),radios=['official','unseen'].map(value=>Object.assign(element(),{value})),bars=Array.from({length:4},()=>element());
  const extra=new Map(['.feature-key','.matrix-axis','.matrix-y'].map(s=>[s,element()]));
  const document={hidden:false,listeners:{},getElementById:id=>nodes.get(id),createElement:()=>element(),createTextNode:text=>({textContent:text}),
    querySelector:s=>extra.get(s),querySelectorAll:s=>s==='[role="tab"][data-stage]'?tabs:s==='[data-layer]'?layers:s==='.bar'?bars:radios,
    addEventListener(k,fn){this.listeners[k]=fn;}};
  const video=nodes.get('exampleVideo');Object.assign(video,{currentTime:0,readyState:2,seeking:false,paused:true,pause(){this.paused=true;},play(){this.paused=false;return Promise.resolve();}});
  const media={matches:reduced,addEventListener(){}};
  const window={devicePixelRatio:1,matchMedia:()=>media,addEventListener(){},speechSynthesis:{getVoices:()=>[{localService:true,lang:'en-AU',name:'Local'},{localService:false,lang:'en-US',name:'Network'}],addEventListener(){},cancel(){},speak(v){speechCalls.push(v);}}};
  vm.runInNewContext(source,{document,window,location:{hostname,search},URLSearchParams,AbortSignal,performance:{now:()=>now},
    requestAnimationFrame:fn=>{const id=nextID++;pending.set(id,fn);return id;},cancelAnimationFrame:id=>pending.delete(id),
    SpeechSynthesisUtterance:function(text){this.text=text;},
    fetch:async url=>{requests.push(url);return {ok:true,json:async()=>url.includes('manifest')?data:url.includes('index.json')?{examples:[{id:'video_75_27'}]}:{}};},console});
  await new Promise(resolve=>setImmediate(resolve));
  return {nodes,tabs,layers,radios,bars,requests,speechCalls,video,document,pending,
    advance(ms){now+=ms;if(!video.paused)video.currentTime=Math.min(video.currentTime+ms/1000,2.96);const callbacks=[...pending.values()];pending.clear();callbacks.forEach(fn=>fn(now));}};
}

test('explicit schematic mode has no dataset request or autoplay',async()=>{
  const h=await harness({search:'?example=schematic'});assert.deepEqual(h.requests,[]);assert.equal(h.nodes.get('speakButton').disabled,true);assert.equal(h.pending.size,0);assert.equal(h.speechCalls.length,0);
});
test('local-example query on a public host still uses the public random catalogue',async()=>{
  const h=await harness({hostname:'auslan-research.vercel.app',search:'?example=local'});assert.deepEqual(h.requests,['/static/examples/index.json','/static/examples/video_75_27/manifest.json']);assert.equal(h.video.src,'/static/examples/video_75_27/clip.mp4');
});
test('recorded replay scrubs video and pose to the same frame',async()=>{
  const h=await harness({search:'?example=local'});const timeline=h.nodes.get('timeline');timeline.value='500';timeline.emit('input');
  assert.equal(h.video.currentTime,1.48);assert.equal(h.nodes.get('frameLabel').textContent,'FRAME 038 / 74');assert.equal(h.nodes.get('referenceText').textContent,'a fixture .');
  h.tabs[2].emit('click');h.layers[5].emit('click');assert.match(h.nodes.get('featureCanvas').attributes['aria-label'],/Encoder 4 measured feature matrix/);
});
test('invalid trace falls back visibly, without showing an invented prediction',async()=>{
  const data=fixture();data.points.pop();const h=await harness({search:'?example=local',data});
  assert.equal(h.nodes.get('exampleError').hidden,false);assert.equal(h.nodes.get('referenceBlock').hidden,true);assert.equal(h.nodes.get('exampleBadge').textContent,'Interactive schematic');
});
test('keyboard navigation selects stages; speech always needs its own button',async()=>{
  const h=await harness({search:'?example=local'});h.tabs[0].emit('keydown',{key:'End',preventDefault(){}});
  assert.equal(h.tabs[4].attributes['aria-selected'],'true');assert.equal(h.tabs[4].focused,true);assert.equal(h.speechCalls.length,0);
  h.nodes.get('speakButton').emit('click');assert.equal(h.speechCalls.length,1);assert.equal(h.speechCalls[0].voice.localService,true);
});
test('withheld output cannot be spoken',async()=>{
  const h=await harness({search:'?example=local',data:fixture(false)});h.tabs[4].emit('click');assert.equal(h.nodes.get('speakButton').disabled,true);
  h.nodes.get('speakButton').emit('click');assert.equal(h.speechCalls.length,0);
});
test('guided tour uses one animation loop and ends silently on the speech stage',async()=>{
  const h=await harness({search:'?example=local'});h.nodes.get('tourButton').emit('click');
  for(let i=0;i<220;i++){h.advance(100);assert.ok(h.pending.size<=1,'at most one animation callback');}
  assert.equal(h.tabs[4].attributes['aria-selected'],'true');assert.equal(h.pending.size,0);assert.equal(h.speechCalls.length,0);
});
test('reduced motion uses manual progression; backgrounding stops playback',async()=>{
  const h=await harness({reduced:true});h.nodes.get('tourButton').emit('click');assert.equal(h.tabs[1].attributes['aria-selected'],'true');assert.equal(h.pending.size,0);
  h.nodes.get('playButton').emit('click');assert.equal(h.pending.size,1);h.document.hidden=true;h.document.listeners.visibilitychange();assert.equal(h.pending.size,0);
});



test('feature explanation follows the selected point and frame; inspection preserves actual channels',async()=>{
  const h=await harness({search:'?example=local'});h.tabs[2].emit('click');
  const timeline=h.nodes.get('timeline');timeline.value='500';timeline.emit('input');
  const point=h.nodes.get('featurePoint');point.value='112';point.emit('change');
  assert.equal(h.nodes.get('coordinateX').textContent,'0.706');
  assert.equal(h.nodes.get('featureFrameLabel').textContent,'FRAME 038 · 1.48 s');
  h.nodes.get('featureInspect').emit('click');
  assert.equal(h.nodes.get('featureInspection').hidden,false);
  const channel=h.nodes.get('featureChannel');channel.value='398';channel.emit('input');
  assert.equal(h.nodes.get('featureValue').textContent,'0.76800');
  h.layers[1].emit('click');assert.equal(channel.value,'255');
  assert.equal(h.nodes.get('featureValue').textContent,'0.62500');
  h.nodes.get('featureExplain').emit('click');assert.equal(h.nodes.get('featureInspection').hidden,true);
  assert.match(h.nodes.get('featureFlowCanvas').attributes['aria-label'],/All 256 projected/);
});
test('attention is selected by layer and query frame, not fabricated from features',async()=>{
  const data=fixture();data.attention[2][37]=Array(74).fill(0);data.attention[2][37][9]=1;
  const h=await harness({search:'?example=local',data});h.tabs[2].emit('click');h.layers[4].emit('click');
  h.nodes.get('timeline').value='500';h.nodes.get('timeline').emit('input');
  assert.match(h.nodes.get('featureFlowCanvas').attributes['aria-label'],/Encoder layer 3 self-attention from frame 38/);
  assert.match(h.nodes.get('featureFlowCaption').textContent,/frame 10 \(100.0%\)/);
});
test('matrix selection updates frame and channel, with keyboard controls offering the same values',async()=>{
  const h=await harness({search:'?example=local'});h.tabs[2].emit('click');h.layers[2].emit('click');h.nodes.get('featureInspect').emit('click');
  h.nodes.get('featureCanvas').emit('click',{clientX:200,clientY:150});
  assert.equal(h.video.currentTime,1.48);assert.equal(h.nodes.get('featureChannel').value,'128');
  assert.equal(h.nodes.get('featureValue').textContent,'0.49800');
});
test('schematic explains layers without fabricated activations or attention',async()=>{
  const h=await harness({search:'?example=schematic'});h.tabs[2].emit('click');h.layers[2].emit('click');
  assert.equal(h.nodes.get('featureInspect').disabled,true);
  assert.match(h.nodes.get('featureFlowCanvas').attributes['aria-label'],/Architecture only; no measured values/);
  assert.equal(h.nodes.get('featureFlowCaption').textContent.includes('not measured weights'),true);
});
test('invalid attention distributions and channel shapes reject the recorded trace',async()=>{
  for(const corrupt of [d=>{d.attention[0][0][0]=-1;},d=>{d.attention[0][0].fill(0);},d=>{d.features[1].values[0].pop();}]){
    const data=fixture();corrupt(data);const h=await harness({search:'?example=local',data});
    assert.equal(h.nodes.get('exampleError').hidden,false);assert.equal(h.nodes.get('referenceBlock').hidden,true);
  }
});

test('public landing page loads a catalogue-selected human video and trace by default, without autoplay',async()=>{
  const h=await harness({hostname:'auslan-research.vercel.app'});
  assert.deepEqual(h.requests,['/static/examples/index.json','/static/examples/video_75_27/manifest.json']);
  assert.equal(h.video.src,'/static/examples/video_75_27/clip.mp4');
  assert.equal(h.nodes.get('videoPane').hidden,false);assert.equal(h.nodes.get('exampleCredit').hidden,false);
  assert.equal(h.video.paused,true);assert.equal(h.pending.size,0);assert.equal(h.speechCalls.length,0);
});
test('public media paths cannot be changed by an untrusted manifest',async()=>{
  const data=fixture();data.video='https://example.com/unapproved.mp4';
  const h=await harness({hostname:'auslan-research.vercel.app',data});
  assert.equal(h.nodes.get('exampleError').hidden,false);assert.equal(h.video.src,undefined);
  assert.equal(h.nodes.get('exampleCredit').hidden,true);
});
