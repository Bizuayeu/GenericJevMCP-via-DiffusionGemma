import assert from 'node:assert/strict';
import {test} from 'node:test';
import {invocation,toolResult,outputSchema} from '../mcp/server.mjs';
test('question and shell metacharacters stay in JSON stdin',()=>{
 const request={questions:{q:{type:'noul',instructions:'魏延？; $(echo injected)'}}};
 const value=invocation({request,image:'a & b.png',state_files:['日本語.md']});
 assert.deepEqual(JSON.parse(value.stdin),request);
 assert.ok(!value.args.some(a=>a.includes('echo injected')));
 assert.deepEqual(value.args.slice(-4),['--image','a & b.png','--state-file','日本語.md']);
});
test('arbitrary commands and malformed input are rejected',()=>{
 for(const input of [null,{}, {request:[]},{request:{},command:'other'}, {request:{},state_files:'file'}, {request:{},image:3}])
   assert.throws(()=>invocation(input));
});

test('MCP returns typed timing and the canonical display without rewriting',()=>{
 const value={answers:{complexity:null},timing:{total_seconds:.882,decision_seconds:.106},display:'所要時間（全体）: 0.882 秒\n判定時間: 0.106 秒'};
 const result=toolResult(JSON.stringify(value));
 assert.deepEqual(result.structuredContent,value);
 assert.equal(result.content[0].text,'```text\n'+value.display+'\n```');
 assert.deepEqual(outputSchema.properties.timing.required,['total_seconds','decision_seconds']);
 assert.throws(()=>toolResult('not JSON'));
 assert.throws(()=>toolResult(JSON.stringify({...value,timing:{total_seconds:.8,decision_seconds:'complexity'}})));
 assert.throws(()=>toolResult(JSON.stringify({...value,display:null})));
});
