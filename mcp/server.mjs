import {fileURLToPath} from 'node:url';
import {spawn} from 'node:child_process';
import {Server} from '@modelcontextprotocol/sdk/server/index.js';
import {StdioServerTransport} from '@modelcontextprotocol/sdk/server/stdio.js';
import {ListToolsRequestSchema,CallToolRequestSchema} from '@modelcontextprotocol/sdk/types.js';
const client=fileURLToPath(new URL('../client.py',import.meta.url));

export function invocation(input) {
  if (!input || typeof input!=='object' || Array.isArray(input) ||
      Object.keys(input).some(k=>!['request','image','state_files'].includes(k)) ||
      !input.request || typeof input.request!=='object' || Array.isArray(input.request))
    throw new Error('request must be a Jev JSON object');
  const args=['-X','utf8',client,'decide','--request-json','-','--format','text'];
  if(input.image!==undefined) {
    if(typeof input.image!=='string' || !input.image) throw new Error('image must be a file path');
    args.push('--image',input.image);
  }
  if(input.state_files!==undefined) {
    if(!Array.isArray(input.state_files) || input.state_files.some(p=>typeof p!=='string'||!p))
      throw new Error('state_files must contain file paths');
    for(const p of input.state_files) args.push('--state-file',p);
  }
  return {args,stdin:JSON.stringify(input.request)};
}
export function decide(input,signal) {
  const {args,stdin}=invocation(input);
  return new Promise((resolve,reject)=>{
    const child=spawn(process.env.JEV_PYTHON || 'python',args,{windowsHide:true,shell:false,signal});
    let output='',errors='';
    child.stdout.setEncoding('utf8'); child.stderr.setEncoding('utf8');
    child.stdout.on('data',s=>output+=s); child.stderr.on('data',s=>errors+=s);
    child.on('error',reject);
    child.stdin.on('error',()=>{}); // A failed spawn/early exit is handled below.
    child.on('close',code=>code===0?resolve(output):reject(new Error(errors.trim()||'Jev client failed')));
    child.stdin.end(stdin);
  });
}
export async function serve() {
  const server=new Server({name:'jev',version:'0.1.0'},{capabilities:{tools:{}}});
  server.setRequestHandler(ListToolsRequestSchema,async()=>({tools:[{
    name:'decide',description:'Return yes/no (noul), choice or score decisions and elapsed seconds using DiffusionGemma. request.questions maps IDs to {type, instructions, criteria}; choice criteria map names to descriptions or null; score criteria are ordered labels. state is optional evidence; sources_only:true restricts evidence. image/state_files read local files and send them to the configured decision server. Files are not modified. Probabilities are uncalibrated and normalized within the supplied candidates, not factual correctness probabilities. Explain this when asked rather than repeating a disclaimer in every result.',
    inputSchema:{type:'object',properties:{request:{type:'object',properties:{questions:{type:'object'},state:{},rag:{},mode:{type:'string'},samples:{},sources_only:{type:'boolean'},seed:{type:'integer'}},required:['questions']},image:{type:'string'},state_files:{type:'array',items:{type:'string'}}},required:['request'],additionalProperties:false},
    annotations:{readOnlyHint:true,destructiveHint:false,openWorldHint:true}
  }]}));
  server.setRequestHandler(CallToolRequestSchema,async(request,extra)=>{
    try {
      if(request.params.name!=='decide') throw new Error('Unknown tool');
      return {content:[{type:'text',text:await decide(request.params.arguments,extra.signal)}]};
    } catch(error) {
      return {isError:true,content:[{type:'text',text:error.message}]};
    }
  });
  await server.connect(new StdioServerTransport());
}
if(process.argv[1] && fileURLToPath(import.meta.url)===process.argv[1]) await serve();
