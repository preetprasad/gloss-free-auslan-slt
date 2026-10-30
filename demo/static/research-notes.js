(() => {
  'use strict';
  const $=id=>document.getElementById(id);
  const text=(id,value)=>{$(id).textContent=value;};
  const values={official:[9.522,10.272,10.335,11.802],unseen:[4.88,5.31,5.81,6.60]};
  const labels=['Movement only','With extra pretraining','Movement and images','Six models combined'];
  document.querySelectorAll('input[name="subset"]').forEach(input=>input.addEventListener('change',()=>{
    document.querySelectorAll('.bar').forEach((bar,i)=>{bar.style.width=values[input.value][i]/12.5*100+'%';bar.querySelector('b').textContent=values[input.value][i].toFixed(2);});
    $('resultsChart').setAttribute('aria-label',(input.value==='unseen'?'Unseen English targets':'Official test')+' BLEU-4: '+labels.map((label,i)=>label+' '+values[input.value][i].toFixed(2)).join(', ')+'.');
    text('chartCaption',input.value==='unseen'?'570 clips with English reference sentences absent from training. This tests new sentences, not new signers.':'800 recorded test clips. The combined system uses three movement-only translators and three that also see images.');
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
})();
