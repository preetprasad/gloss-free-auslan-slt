'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.join(__dirname, '../static');
const source = fs.readFileSync(path.join(root, 'research-notes.js'), 'utf8');
const html = fs.readFileSync(path.join(root, 'research-notes.html'), 'utf8');

async function harness(fail = false) {
  const element = () => ({textContent: '', attributes: {}, children: [], style: {},
    addEventListener(event, fn) { this[event] = fn; },
    setAttribute(key, value) { this.attributes[key] = value; },
    append(child) { this.children.push(child); },
    querySelector() { return this.label ||= element(); }});
  const nodes = new Map([...html.matchAll(/id="([^"]+)"/g)].map(m => [m[1], element()]));
  const radios = ['official', 'unseen'].map(value => Object.assign(element(), {value}));
  const bars = Array.from({length: 4}, element);
  const snapshot = JSON.parse(fs.readFileSync(path.join(root, 'evidence.json'), 'utf8'));
  vm.runInNewContext(source, {
    document: {getElementById: id => nodes.get(id), createElement: element,
      querySelectorAll: selector => selector === '.bar' ? bars : radios},
    fetch: async () => { if (fail) throw Error('offline'); return {ok: true, json: async () => snapshot}; },
  });
  await new Promise(resolve => setImmediate(resolve));
  return {nodes, radios, bars};
}

test('research page switches subset scores and accessible context without the walkthrough controller', async () => {
  const h = await harness();
  h.radios[1].change();
  assert.equal(h.bars[3].label.textContent, '6.60');
  assert.match(h.nodes.get('resultsChart').attributes['aria-label'], /Unseen English targets/);
  assert.match(h.nodes.get('chartCaption').textContent, /not new signers/);
  h.radios[0].change();
  assert.equal(h.bars[3].label.textContent, '11.80');
  assert.match(h.nodes.get('chartCaption').textContent, /800 recorded test clips/);
  assert.ok(h.nodes.get('liveEvidence').children.length > 0);
});

test('unavailable evidence snapshot does not break the static page or subset control', async () => {
  const h = await harness(true);
  assert.equal(h.nodes.get('liveEvidence').children.length, 0);
  h.radios[1].change();
  assert.equal(h.bars[0].label.textContent, '4.88');
});
