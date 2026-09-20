import assert from 'node:assert/strict';
import {test} from 'node:test';
import {invocation} from '../mcp/server.mjs';
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
