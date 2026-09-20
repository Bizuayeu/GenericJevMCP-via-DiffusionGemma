import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
import {Client} from '@modelcontextprotocol/sdk/client/index.js';
import {StdioClientTransport} from '@modelcontextprotocol/sdk/client/stdio.js';

const client=new Client({name:'jev-smoke',version:'0.1.0'});
const transport=new StdioClientTransport({
  command:process.execPath,
  args:[fileURLToPath(new URL('../../mcp/server.mjs',import.meta.url))],
  env:Object.fromEntries(Object.entries(process.env).filter(([,v])=>v!==undefined)),
  stderr:'inherit',
});
try {
  await client.connect(transport);
  const listed=await client.listTools();
  assert.ok(listed.tools.some(t=>t.name==='decide'));
  const result=await client.callTool({name:'decide',arguments:{
    request:{state:'The package contains three red balls.',samples:1,questions:{
      red:{type:'noul',instructions:'Does the package contain red balls?'},
      color:{type:'choice',instructions:'What color are the balls?',criteria:{red:null,blue:null}},
      explicitness:{type:'score',instructions:'How explicitly is the color stated?',criteria:['not stated','implied','explicit']}
    }}
  }},undefined,{timeout:240000});
  assert.ok(!result.isError,JSON.stringify(result));
  const output=result.content.filter(c=>c.type==='text').map(c=>c.text).join('\n');
  assert.match(output,/red:/);
  assert.match(output,/color:/);
  assert.match(output,/explicitness:/);
  assert.match(output,/所要時間（全体）:/);
  assert.equal(output,'```text\n'+result.structuredContent.display+'\n```');
  assert.equal(typeof result.structuredContent.timing.total_seconds,'number');
  assert.equal(typeof result.structuredContent.timing.decision_seconds,'number');
  assert.match(output,/判定時間:/);
  assert.match(output,/confidence:/);
  assert.match(output,/entropy:/);
  assert.equal(typeof result.structuredContent.metrics.color.confidence,'number');
  assert.equal(typeof result.structuredContent.metrics.color.entropy_nats,'number');
  console.log(output);
} finally {await client.close();}
