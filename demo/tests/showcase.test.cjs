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
  return {version:1,kind:'recorded',clip:'test-fixture',frames:74,fps:25,duration:2.96,width:778,height:838,
    video:'clip.mp4',poster:'poster.jpg',points:Array.from({length:74},()=>Array.from({length:133},()=>[1,2,.9])),
    features:Array.from({length:6},(_,i)=>({width:i?256:399,values:Array.from({length:74},()=>Array(24).fill(.1))})),
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
      getBoundingClientRect(){return {width:400,height:300};},getContext(){return context2d;},
      focus(){this.focused=true;},querySelector(){return this.valueNode ||= element();}};
  }
  for(const match of html.matchAll(/id="([^"]+)"/g))nodes.set(match[1],element(match[1]));
  for(const id of ['referenceBlock','exampleError','stopSpeech'])nodes.get(id).hidden=true;
  const tabs=Array.from({length:5},(_,i)=>nodes.get('tab-'+i));
  const layers=Array.from({length:6},()=>element()),radios=['official','unseen'].map(value=>Object.assign(element(),{value})),bars=Array.from({length:4},()=>element());
  const extra=new Map(['.feature-key','.matrix-axis','.matrix-y'].map(s=>[s,element()]));
  const document={hidden:false,listeners:{},getElementById:id=>nodes.get(id),createElement:()=>element(),createTextNode:text=>({textContent:text}),
    querySelector:s=>extra.get(s),querySelectorAll:s=>s==='[role="tab"][data-stage]'?tabs:s==='[data-layer]'?layers:s==='.bar'?bars:radios,
    addEventListener(k,fn){this.listeners[k]=fn;}};
  const video=nodes.get('exampleVideo');Object.assign(video,{currentTime:0,paused:true,pause(){this.paused=true;},play(){this.paused=false;return Promise.resolve();}});
  const media={matches:reduced,addEventListener(){}};
  const window={devicePixelRatio:1,matchMedia:()=>media,addEventListener(){},speechSynthesis:{getVoices:()=>[{localService:true,lang:'en-AU',name:'Local'},{localService:false,lang:'en-US',name:'Network'}],addEventListener(){},cancel(){},speak(v){speechCalls.push(v);}}};
  vm.runInNewContext(source,{document,window,location:{hostname,search},URLSearchParams,AbortSignal,performance:{now:()=>now},
    requestAnimationFrame:fn=>{const id=nextID++;pending.set(id,fn);return id;},cancelAnimationFrame:id=>pending.delete(id),
    SpeechSynthesisUtterance:function(text){this.text=text;},
    fetch:async url=>{requests.push(url);return {ok:true,json:async()=>url.includes('manifest')?data:{}};},console});
  await new Promise(resolve=>setImmediate(resolve));
  return {nodes,tabs,layers,radios,bars,requests,speechCalls,video,document,pending,
    advance(ms){now+=ms;if(!video.paused)video.currentTime=Math.min(video.currentTime+ms/1000,2.96);const callbacks=[...pending.values()];pending.clear();callbacks.forEach(fn=>fn(now));}};
}

test('public default is an explicit schematic, with no dataset request or autoplay',async()=>{
  const h=await harness();assert.deepEqual(h.requests,['/static/evidence.json']);assert.equal(h.nodes.get('speakButton').disabled,true);assert.equal(h.pending.size,0);assert.equal(h.speechCalls.length,0);
});
test('local-example query is ignored on public hosts',async()=>{
  const h=await harness({hostname:'auslan-research.vercel.app',search:'?example=local'});assert.deepEqual(h.requests,['/static/evidence.json']);
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
test('subset comparison updates both values and accessible chart description',async()=>{
  const h=await harness();h.radios[1].emit('change');assert.equal(h.bars[3].valueNode.textContent,'6.60');assert.match(h.nodes.get('resultsChart').attributes['aria-label'],/Unseen English targets/);
});
