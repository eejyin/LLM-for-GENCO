'use strict';
const $ = id => document.getElementById(id);
const GROUPS = {data:'Inputs & data',projection:'Feature projection',correction:'Refinement loop',loss:'Training loss',optimizer:'Optimization & early stopping'};
const DETAILS = {
 data:'Buses, generators, and branches form a heterogeneous graph. Inputs retain known states; task transforms handle unknown quantities. Train, validation, and test splits are separate. Fix the data split before comparing experiments.',
 projection:'Separate MLPs project buses, generators, and branches. hidden_size = H sets representation width. Branch embeddings remain fixed during refinement; node embeddings keep updating.\nSource: models/gnn_heterogeneous_gns.py · input_proj_bus / gen / edge',
 correction:'Each round uses HeteroConv + TransformerConv to aggregate three relation types. attention_head sets parallel attention heads; num_layers sets the number of refinements. The post-attention width is H × heads. Residual encoding feeds the next round.\nSource: models/gnn_heterogeneous_gns.py · layers / physics_mlp',
 decoder:'A shared MLP decodes voltage states; OPF also decodes generator active power. The physics decoder uses admittance to reconstruct branch flows and injections. In the paper, OPF uses sigmoid box constraints; PF and SE do not. The YAML determines the task.',
 loss:'Mixed losses pair training.losses with loss_weights in order. LayeredWeightedPhysics combines residuals using normalized geometric weights. Smaller λ emphasizes later stages; the corresponding loss_weight sets the total physics-loss strength.\nSource: training/loss.py · LayeredWeightedPhysicsLoss / MixedLoss',
 optimizer:'Forward refinement produces states and losses. Backpropagation computes gradients; the optimizer updates projection, attention, decoder, and residual-encoder weights. learning_rate controls update size, batch_size controls graphs per update, and epochs caps full passes through training data. Validation metrics guide early stopping and checkpoints.'
};
let description=null, previewVersion=0, previewTimer=null, selectedGroup=null, selectedJob=null, jobState=null, planState=null, runList=[], plotSeries=[], pendingBenchmarks=new Map(), caseGeneration=0;
let chatBusy=false, planBusy=false, autoPlan=false, undoSnapshot=null, parametersValid=true, setupReady=false;
let conversationHistory=[];
const TERMINAL=['completed','failed','cancelled'];
const ROUTE_NAMES={QUESTION_GENERAL:'Architecture Q&A',QUESTION_PARAMETER:'Parameter Q&A',PARAMETER:'Change parameters',GENERATION:'Start training',ANALYSIS:'Analyze results',PLANNER:'Plan experiment'};
const colors=['#287350','#d49a3e','#476bb2','#ab62a3','#df7561'];
async function api(url, body) {
 const response=await fetch(url,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 let data;try{data=await response.json()}catch{throw Error('Server did not return JSON')}
 if(!response.ok)throw Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail||response.status));
 return data;
}
function say(text,kind='bot'){const el=document.createElement('div');el.className='message '+kind;el.textContent=text;$('chat').append(el);$('chat').scrollTop=$('chat').scrollHeight;if(kind!=='error'&&text){conversationHistory.push({role:kind==='user'?'user':'assistant',content:String(text).slice(0,2000)});conversationHistory=conversationHistory.slice(-6)}}
function notify(text,error=false){$('notification-text').textContent=text;$('notification').classList.toggle('error',error);$('notification').hidden=false}
function problem(error){const text=error.message||String(error);say(text,'error');notify(text,true)}
async function busy(button,fn){if(button.dataset.busy)return;button.dataset.busy='true';button.disabled=true;button.setAttribute('aria-busy','true');try{return await fn()}catch(e){problem(e)}finally{delete button.dataset.busy;button.removeAttribute('aria-busy');button.disabled=false;syncControls()}}
function syncControls(){
 const locked=chatBusy||planBusy;
 $('send').disabled=locked||!setupReady;$('route-mode').disabled=locked||!setupReady;$('chat-source').disabled=locked||!setupReady;
 document.querySelectorAll('[data-prompt]').forEach(b=>b.disabled=locked||!setupReady);
 $('advance').disabled=locked||autoPlan||!planState||planState.status!=='ready'||!!planState.waiting_job;
 $('auto-plan').disabled=locked||!planState||TERMINAL.includes(planState.status);
 $('stop-plan').disabled=locked||!planState||TERMINAL.includes(planState.status);
 $('auto-plan').textContent=autoPlan?'Pause automatic execution':'Run remaining steps';
 $('train').disabled=!setupReady||!parametersValid||locked||!!$('train').dataset.busy||!!(planState&&!TERMINAL.includes(planState.status));
 $('preflight').disabled=!setupReady||!!$('preflight').dataset.busy;
 $('tour-training').disabled=!setupReady;
 $('export-config').disabled=!setupReady;
 $('undo-parameters').disabled=!undoSnapshot||locked||!!(planState&&!TERMINAL.includes(planState.status));
 document.querySelectorAll('[data-key], [data-width], #config, #reset, #data-path').forEach(el=>el.disabled=!setupReady||locked||!!(planState&&!TERMINAL.includes(planState.status))||(el.dataset.width!==undefined&&!description?.architecture_supported));
}
function tab(name){document.querySelectorAll('[data-tab]').forEach(b=>b.classList.toggle('active',b.dataset.tab===name));document.querySelectorAll('.tab').forEach(s=>s.classList.toggle('active',s.id===name+'-tab'));requestAnimationFrame(redraw)}
function download(name,text,type='text/plain'){const a=document.createElement('a'),url=URL.createObjectURL(new Blob([text],{type}));a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
function fieldValues(){const result={};document.querySelectorAll('[data-key]').forEach(input=>{if(input.value.trim()===''||!Number.isFinite(Number(input.value)))throw Error(input.dataset.key+' requires a finite number');result[input.dataset.key]=Number(input.value)});return result}
function trainingRequest(){return {config:$('config').value,data_path:$('data-path').value,values:fieldValues()}}
function chatRequest(message){return {...trainingRequest(),message,history:conversationHistory.slice(-6),route:$('route-mode').value||null,case:$('case').value,job_id:$('chat-source').value==='case'?null:$('chat-source').value}}
function selectGroup(group){selectedGroup=group;document.querySelectorAll('.stage').forEach(b=>b.classList.toggle('selected',b.dataset.group===group));document.querySelectorAll('.parameter-group').forEach(d=>{const match=d.dataset.group===group;d.classList.toggle('highlight',match);if(match)d.open=true});$('stage-detail').textContent=DETAILS[group]||''}
function renderFields(info){
 $('parameter-fields').replaceChildren();
 Object.entries(GROUPS).forEach(([group,title])=>{
  const fields=info.fields.filter(f=>f.group===group);if(!fields.length)return;
  const details=document.createElement('details');details.className='parameter-group';details.dataset.group=group;details.open=['projection','correction','loss'].includes(group);
  const summary=document.createElement('summary');summary.textContent=title;details.append(summary);
  for(const f of fields){const div=document.createElement('div');div.className='parameter-field';const label=document.createElement('label');label.textContent=f.key;const input=document.createElement('input');input.type='number';input.step=f.integer?'1':'any';input.min=f.min;input.max=f.max;input.value=f.value;input.dataset.key=f.key;input.setAttribute('aria-label',f.key);input.oninput=()=>{clearTimeout(previewTimer);previewTimer=setTimeout(()=>refreshPreview().catch(problem),250)};label.append(input);const path=document.createElement('code');path.textContent=f.yaml_path;const help=document.createElement('p');help.textContent=f.help;div.append(label,path,help);details.append(div)}
  $('parameter-fields').append(details);
 });
 document.querySelectorAll('[data-width]').forEach(b=>b.disabled=!info.architecture_supported);
 if(selectedGroup)selectGroup(selectedGroup);
}
function renderArchitecture(info){
 description=info;
 $('config-meta').textContent=info.task+' · '+info.model+' · '+info.networks.join(', ')+' · scenarios: '+info.scenarios.join(', ');
 $('task-name').textContent=info.task;
 $('projection-summary').textContent='H = '+(info.values.hidden_size??'—')+' · node / branch projection';
 $('correction-summary').textContent='N = '+(info.values.num_layers??'—')+' · heads = '+(info.values.attention_head??'—');
 $('architecture-note').textContent=info.architecture_supported?'This configuration uses the local GNS_heterogeneous GENCO implementation.':'Selected model: '+info.model+'; the diagram is a GENCO reference and may not match the selected model.';
 $('yaml-preview').textContent=info.yaml;$('code-source').textContent='Local source: '+info.sources.join(' · ');
 const weights=info.layer_weights;const max=Math.max(...weights,1e-12);
 $('weight-bars').replaceChildren(...weights.map((w,i)=>{const div=document.createElement('div');div.className='weight-bar';div.title='Refinement stage '+(i+1)+' · '+(w*100).toFixed(4)+'%';const bar=document.createElement('i');bar.style.height=(w/max*100)+'%';const label=document.createElement('span');label.textContent=i+1;div.append(bar,label);return div}));
 $('weight-note').textContent=weights.length?'λ = '+info.values.base_weight+' · '+weights.length+' stages · weights sum to 1':'LayeredWeightedPhysics is not enabled in this configuration';
}
async function loadParameters(){const v=++previewVersion;const info=await api('/api/parameters/'+encodeURIComponent($('config').value));if(v!==previewVersion)return;undoSnapshot=null;renderFields(info);renderArchitecture(info);$('validation').textContent='Configuration loaded. Changes update the architecture preview and YAML.';$('validation').className='notice';parametersValid=true;syncControls();saveDraft()}
async function refreshPreview(){const version=++previewVersion;try{const info=await api('/api/parameters/preview',trainingRequest());if(version!==previewVersion)return;renderArchitecture(info);$('validation').textContent='Valid parameters · This run will use the current YAML snapshot.';$('validation').className='notice';parametersValid=true;syncControls();saveDraft();return info}catch(e){if(version===previewVersion){$('validation').textContent=e.message;$('validation').className='notice error';parametersValid=false;syncControls()}throw e}}
function applyChanges(changes){for(const [key,value] of Object.entries(changes||{})){const input=[...document.querySelectorAll('[data-key]')].find(i=>i.dataset.key===key);if(input)input.value=value}}
function showPlan(plan){planState=plan;if(TERMINAL.includes(plan.status))autoPlan=false;$('plan').hidden=false;$('plan-steps').replaceChildren(...plan.steps.map((step,i)=>{const li=document.createElement('li');const waiting=!!plan.waiting_job&&i===plan.index-1;li.textContent=(waiting?'◷ ':i<plan.index?'✓ ':i===plan.index?'→ ':'')+(ROUTE_NAMES[step.route]||step.route)+' · '+(step.instruction||'')+(Object.keys(step.changes||{}).length?' · '+JSON.stringify(step.changes):'');li.classList.toggle('done',i<plan.index&&!waiting);li.classList.toggle('current',i===plan.index||waiting);li.classList.toggle('failed',plan.status==='failed'&&(waiting||i===plan.index));return li}));const status={ready:'Ready for next step',waiting:'Waiting for training',executing:'Executing step',completed:'Completed',failed:'Failed',cancelled:'Stopped'};$('plan-status').textContent=(status[plan.status]||plan.status)+' · '+plan.index+'/'+plan.steps.length+(plan.error?' · '+plan.error:'');syncControls();saveDraft()}
async function handleRoute(result){$('route').textContent=(result.trace||[result.route,result.next]).join(' → ');say(result.reply,result.error?'error':'bot');if(result.error)notify(result.reply+' '+(result.recovery||''),true);const changes=result.changes||{};if(Object.keys(changes).length){undoSnapshot={config:$('config').value,values:fieldValues()};const before=fieldValues();applyChanges(changes);$('parameter-diff').hidden=false;$('parameter-diff').replaceChildren(...Object.entries(changes).map(([key,value])=>{const span=document.createElement('span');span.textContent=key+': '+before[key]+' → '+value;return span}));await refreshPreview();notify('Parameters updated. Inspect them in the panel or undo the change.')}if(result.plan)showPlan(result.plan);if(result.job){await refreshJobs();await selectJob(result.job.id);tab('results');notify('Training job submitted: '+result.job.id)}if(result.report)showReport(result.report);syncControls()}

function saveDraft(){try{if(!description)return;localStorage.setItem('genco-draft-v1',JSON.stringify({config:$('config').value,values:fieldValues(),data_path:$('data-path').value,plan_id:planState&&!TERMINAL.includes(planState.status)?planState.id:null,message:$('message').value,route:$('route-mode').value}))}catch{}}
async function checkTraining(){const result=await api('/api/train/preflight',trainingRequest());$('preflight-result').hidden=false;$('preflight-result').className='notice'+(result.ready?'':' error');$('preflight-result').textContent=(result.ready?'Preflight passed; the run log remains authoritative.':result.issues.join('\n'))+'\nPython: '+result.python+'\n'+result.hint;if(!result.ready)notify('Training is not ready: '+result.issues.join('; '),true);else notify('Training environment and input data passed preflight.');return result}
async function advancePlan(){if(planBusy||!planState||planState.status!=='ready')return;planBusy=true;syncControls();try{await handleRoute(await api('/api/plans/'+planState.id+'/advance',{}))}catch(e){autoPlan=false;problem(e)}finally{planBusy=false;syncControls()}}
async function refreshPlan(){if(!planState||planBusy||chatBusy||TERMINAL.includes(planState.status))return;try{const plan=await api('/api/plans/'+planState.id);if(plan.status!==planState.status){showPlan(plan);if(plan.status==='failed')notify(plan.error,true);if(plan.status==='completed')notify('Experiment plan completed.')}if(autoPlan&&plan.status==='ready')await advancePlan()}catch(e){autoPlan=false;planState.status='failed';planState.error=e.message;showPlan(planState);problem(e)}}
function draw(canvas,series,xLabel,yLabel){
 const w=canvas.clientWidth,h=canvas.clientHeight;if(w<10||h<10)return;
 const dpr=window.devicePixelRatio||1;canvas.width=w*dpr;canvas.height=h*dpr;const g=canvas.getContext('2d');g.scale(dpr,dpr);g.clearRect(0,0,w,h);
 const all=series.flatMap(s=>s.points).filter(p=>Number.isFinite(p.x)&&Number.isFinite(p.y));
 if(!all.length){g.fillStyle='#65786f';g.font='12px sans-serif';g.fillText('No recorded data to plot',30,h/2);return}
 let xmin=Math.min(...all.map(p=>p.x)),xmax=Math.max(...all.map(p=>p.x)),ymin=Math.min(...all.map(p=>p.y)),ymax=Math.max(...all.map(p=>p.y));const padding=(ymax-ymin)*.12||.01;ymin-=padding;ymax+=padding;if(xmax===xmin)xmax=xmin+1;
 const x=v=>53+(v-xmin)/(xmax-xmin)*(w-72),y=v=>h-30-(v-ymin)/(ymax-ymin)*(h-55);
 g.font='10px sans-serif';
 for(let i=0;i<=4;i++){const value=ymin+(ymax-ymin)*i/4,yy=y(value);g.strokeStyle='#e1e9df';g.beginPath();g.moveTo(53,yy);g.lineTo(w-19,yy);g.stroke();g.fillStyle='#65786f';g.fillText(Math.abs(value)<.001&&value!==0?value.toExponential(1):value.toFixed(3),0,yy+3)}
 for(let i=0;i<=4;i++){const v=xmin+(xmax-xmin)*i/4;g.fillText(String(Math.round(v)),x(v)-6,h-15)}
 g.fillText(yLabel,3,12);g.fillText(xLabel,w-70,h-2);
 for(const s of series){if(!s.points.length)continue;g.strokeStyle=s.color;g.lineWidth=1.8;g.setLineDash(s.dashed?[5,4]:[]);g.beginPath();s.points.forEach((p,i)=>i?g.lineTo(x(p.x),y(p.y)):g.moveTo(x(p.x),y(p.y)));g.stroke();if(s.points.length===1){g.fillStyle=s.color;g.beginPath();g.arc(x(s.points[0].x),y(s.points[0].y),3,0,Math.PI*2);g.fill()}}g.setLineDash([]);
}
function redraw(){draw($('plot'),plotSeries.filter(s=>s.visible).map(s=>({...s,points:s.buses.map(b=>({x:b.bus,y:b.vm}))})),'Bus ID','Vm / p.u.');const metric=jobState?.curves?.[Number($('metric-select').value)];draw($('loss-plot'),metric?[{points:metric.points.map(p=>({x:p.step,y:p.value})),color:colors[0]}]:[],'Step','Metric')}
function legend(){const root=$('plot-legend');root.replaceChildren(...plotSeries.map(s=>{const button=document.createElement('button');button.classList.toggle('off',!s.visible);const dot=document.createElement('i');dot.style.background=s.color;button.append(dot,document.createTextNode(s.name));button.onclick=()=>{s.visible=!s.visible;legend();redraw()};return button}))}
function addSeries(name,buses){const existing=plotSeries.find(s=>s.name===name);if(existing){existing.buses=buses;existing.visible=true}else plotSeries.push({name,buses:[...buses].sort((a,b)=>a.bus-b.bus),visible:true,color:colors[plotSeries.length%colors.length],dashed:plotSeries.length>0});legend();redraw()}
async function loadCase(){const token=++caseGeneration;const data=await api('/api/cases/'+encodeURIComponent($('case').value));if(token!==caseGeneration)return;plotSeries=[];addSeries('DataKit AC · saved',data.buses);const s=data.summary;$('vm').textContent=Number(s.vm_min_pu).toFixed(5);$('va').textContent=Number(s.vm_max_pu).toFixed(5);$('pb').textContent=Number(s.max_p_mismatch_mw).toExponential(2);$('source').textContent=data.kind+' · '+$('case').value+' · not GENCO inference';$('benchmark-results').replaceChildren()}
function updateOptions(id,items,emptyLabel){const select=$(id),previous=select.value;select.replaceChildren(...(emptyLabel?[new Option(emptyLabel,'')]:[]),...items.map(j=>new Option(j.id+' · '+j.kind+' · '+j.status,j.id)));if(items.some(i=>i.id===previous))select.value=previous}
async function refreshJobs(){const next=await api('/api/jobs'),signature=items=>items.map(j=>j.id+':'+j.status).join('|');if(signature(next)===signature(runList)&&$('job-select').options.length>1){runList=next;return}runList=next;updateOptions('job-select',runList,'Select a run');if(selectedJob)$('job-select').value=selectedJob;updateOptions('report-run',runList,'Select primary run');updateOptions('compare-run',runList,'No comparison');const source=$('chat-source').value;$('chat-source').replaceChildren(new Option('Saved power-flow case','case'),...runList.map(j=>new Option(j.id+' · '+j.kind+' · '+j.status,j.id)));if([...$('chat-source').options].some(o=>o.value===source))$('chat-source').value=source}
async function selectJob(id){if(!id)return;selectedJob=id;$('job-select').value=id;await refreshSelectedJob()}
async function refreshSelectedJob(){if(!selectedJob)return;const id=selectedJob,data=await api('/api/jobs/'+id);if(id!==selectedJob)return;jobState=data;$('job-status').textContent=id+' · '+data.kind+' · '+data.status+(data.exit_code!==undefined?' · exit '+data.exit_code:'');$('job-log').textContent=data.log_tail||data.error||'No logs yet';$('cancel-job').disabled=!['running','queued'].includes(data.status);const selected=$('metric-select').value;$('metric-select').replaceChildren(...data.curves.map((m,i)=>new Option(m.name,i)));if(Number(selected)<data.curves.length)$('metric-select').value=selected;$('metric-note').textContent=data.curves.length?'Recorded MLflow metrics; x-axis is the logged step.':'No MLflow metrics were recorded for this run; no synthetic curve is shown.';redraw()}
function renderReport(markdown){
 const root=$('report-box');root.replaceChildren();let table=null;
 for(const line of markdown.split('\n')){
  if(!line.trim())continue;
  if(line.startsWith('|')){
   if(/^\|[\s:|\-]+\|$/.test(line))continue;
   const header=!table;if(!table){const wrap=document.createElement('div');wrap.className='report-table';table=document.createElement('table');wrap.append(table);root.append(wrap)}
   const row=document.createElement('tr');for(const cell of line.split('|').slice(1,-1)){const el=document.createElement(header?'th':'td');el.textContent=cell.trim();row.append(el)}table.append(row);continue;
  }
  table=null;const heading=line.match(/^(#{1,3})\s+(.*)$/),el=document.createElement(heading?'h'+Math.min(heading[1].length+1,4):'p');el.textContent=heading?heading[2]:line;root.append(el);
 }
}
function showReport(result){renderReport(result.markdown);$('report-evidence').textContent=JSON.stringify(result.facts,null,2);$('report-download').href=result.download;$('report-download').hidden=false;$('report-warning').hidden=!result.warning;$('report-warning').textContent=result.warning||'';tab('reports');notify(result.warning?'File-based report generated; LLM interpretation is unavailable. You can retry.':'Evidence report generated; you can review and download it.')}
function benchmarkRow(job){const row=document.createElement('div');row.className='benchmark-row';const status=document.createElement('span');status.textContent=job.engine+' · '+job.id+' · '+job.status;const button=document.createElement('button');button.className='secondary';button.textContent='Add to chart';button.disabled=true;row.append(status,button);$('benchmark-results').append(row);pendingBenchmarks.set(job.id,{row,status,button,case:job.case});button.onclick=()=>busy(button,async()=>{const result=await api('/api/jobs/'+job.id+'/power-flow');if(result.case!==$('case').value)throw Error('Select '+result.case+' before adding');addSeries(result.engine+' · '+job.id,result.buses);status.textContent=result.engine+' · '+result.seconds.toFixed(3)+' s (includes conversion/startup)'})}
async function pollBenchmarks(){for(const [id,entry] of pendingBenchmarks){const item=runList.find(j=>j.id===id);if(!item)continue;entry.status.textContent=item.engine+' · '+id+' · '+item.status;if(item.status==='completed'){entry.button.disabled=false;pendingBenchmarks.delete(id)}else if(['failed','cancelled','interrupted'].includes(item.status)){entry.button.textContent='View log';entry.button.disabled=false;entry.button.onclick=()=>selectJob(id).catch(problem);pendingBenchmarks.delete(id)}}}
document.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>tab(b.dataset.tab));
$('tour-results').onclick=()=>{tab('results');document.querySelector('#results-tab article:nth-child(2)').scrollIntoView({behavior:'smooth',block:'start'})};
$('tour-architecture').onclick=()=>{tab('architecture');document.querySelector('.architecture').scrollIntoView({behavior:'smooth',block:'start'});selectGroup('correction')};
$('tour-training').onclick=()=>{document.querySelector('.controls').scrollIntoView({behavior:'smooth',block:'start'});$('preflight').click()};
document.querySelectorAll('.stage').forEach(b=>b.onclick=()=>selectGroup(b.dataset.group));
document.querySelectorAll('[data-width]').forEach(b=>b.onclick=()=>{applyChanges({hidden_size:Number(b.dataset.width)});refreshPreview().catch(problem);selectGroup('projection')});
$('config').onchange=()=>loadParameters().catch(problem);$('reset').onclick=()=>loadParameters().catch(problem);
$('export-config').onclick=()=>busy($('export-config'),async()=>{const d=await refreshPreview();download('genco-config.yaml',d.yaml,'text/yaml')});
$('chat-form').onsubmit=async e=>{
 e.preventDefault();if(chatBusy||planBusy)return;
 const message=$('message').value.trim();if(!message){$('message').focus();return}
 if(planState&&!TERMINAL.includes(planState.status)){notify('Finish or stop the current plan before sending another request.',true);return}
 let body;try{body=chatRequest(message)}catch(e){problem(e);return}
 chatBusy=true;syncControls();say(message,'user');const start=Date.now();
 $('chat-form').closest('.assistant').setAttribute('aria-busy','true');
 const timer=setInterval(()=>$('chat-progress').textContent='Running '+(ROUTE_NAMES[body.route]||'intent detection')+' · '+Math.floor((Date.now()-start)/1000)+' s; the local model may take longer on first load',1000);
 try{const result=await api('/api/chat',body);await handleRoute(result);if(!result.error){$('message').value='';saveDraft()}}
 catch(e){problem(e)}finally{clearInterval(timer);chatBusy=false;$('chat-form').closest('.assistant').removeAttribute('aria-busy');$('chat-progress').textContent='Done · Ctrl / ⌘ + Enter to send';syncControls()}
};
$('advance').onclick=advancePlan;
$('auto-plan').onclick=()=>{autoPlan=!autoPlan;syncControls();if(autoPlan)advancePlan()};
$('stop-plan').onclick=()=>busy($('stop-plan'),async()=>{autoPlan=false;await handleRoute(await api('/api/plans/'+planState.id+'/cancel',{}))});
$('preflight').onclick=()=>busy($('preflight'),checkTraining);
$('train').onclick=()=>busy($('train'),async()=>{await refreshPreview();const check=await checkTraining();if(!check.ready)return;const item=await api('/api/train',trainingRequest());await refreshJobs();await selectJob(item.id);tab('results');say('Training job submitted '+item.id);notify('Training submitted; see Runs & results for logs.')});
$('job-select').onchange=()=>selectJob($('job-select').value).catch(problem);
$('refresh-jobs').onclick=()=>refreshJobs().then(refreshSelectedJob).catch(problem);
$('cancel-job').onclick=()=>busy($('cancel-job'),async()=>{if(selectedJob){await api('/api/jobs/'+selectedJob+'/cancel',{});await refreshSelectedJob();await refreshJobs()}});
$('metric-select').onchange=redraw;$('case').onchange=()=>loadCase().catch(problem);
$('benchmark').onclick=()=>busy($('benchmark'),async()=>{const name=$('case').value,result=await api('/api/benchmark/'+name);if(name!==$('case').value)return;addSeries(result.engine,result.buses)});
document.querySelectorAll('[data-engine]').forEach(button=>button.onclick=()=>busy(button,async()=>{const item=await api('/api/benchmark',{case:$('case').value,engine:button.dataset.engine});benchmarkRow(item);await refreshJobs()}));
$('import-prediction').onclick=()=>busy($('import-prediction'),async()=>{const file=$('prediction-file').files[0];if(!file)throw Error('Select a single-scenario prediction CSV');if(file.size>8_000_000)throw Error('CSV exceeds 8 MB');const name=$('case').value,item=await api('/api/predictions',{case:name,name:'GENCO · '+file.name,csv_text:await file.text()});const result=await api('/api/jobs/'+item.id+'/power-flow');if(name===$('case').value)addSeries(result.engine,result.buses);await refreshJobs();say('Prediction imported. Vm RMSE against the selected saved DataKit scenario: '+result.vm_rmse_vs_saved_datakit.toExponential(4)+' p.u.; verify the scenarios match.')});
$('training-report').onclick=()=>{if(!selectedJob){problem(Error('Select a run first'));return}$('report-type').value='run';$('report-run').value=selectedJob;reportControls();tab('reports');$('report').click()};
function reportControls(){const run=$('report-type').value==='run';for(const id of ['report-run','compare-run'])$(id).parentElement.hidden=!run;for(const id of ['report-case','compare'])$(id).parentElement.hidden=run}
$('report-type').onchange=reportControls;
$('report').onclick=()=>busy($('report'),async()=>{const body=$('report-type').value==='run'?{job_id:$('report-run').value,compare_job_id:$('compare-run').value||null}:{case:$('report-case').value,compare_case:$('compare').value||null};if($('report-type').value==='run'&&!body.job_id)throw Error('Select a primary run');$('report-box').textContent='Generating a report from recorded evidence…';$('report-download').hidden=true;$('report-warning').hidden=true;try{showReport(await api('/api/report',body))}catch(e){$('report-box').textContent='Report not generated: '+e.message;throw e}});
$('save-provider').onclick=()=>busy($('save-provider'),async()=>{const r=await api('/api/provider',{provider:$('provider').value,model:$('model').value});$('model-status').textContent='Applied '+r.provider+' / '+r.model});
window.addEventListener('resize',redraw);
$('dismiss-notification').onclick=()=>$('notification').hidden=true;
$('open-parameters').onclick=()=>document.querySelector('.controls').scrollIntoView({behavior:'smooth',block:'start'});
$('message').onkeydown=e=>{if((e.ctrlKey||e.metaKey)&&e.key==='Enter'){e.preventDefault();$('chat-form').requestSubmit()}};
$('message').oninput=saveDraft;$('data-path').onchange=saveDraft;
document.querySelectorAll('[data-prompt]').forEach(button=>button.onclick=()=>{$('message').value=button.dataset.prompt;$('route-mode').value=button.dataset.route;$('message').focus();saveDraft()});
$('undo-parameters').onclick=async()=>{if(!undoSnapshot)return;const saved=undoSnapshot;undoSnapshot=null;if(saved.config!==$('config').value)return;applyChanges(saved.values);await refreshPreview();$('parameter-diff').hidden=true;notify('Previous parameter values restored.');syncControls()};
async function init(){
 try{
  const [h,c]=await Promise.all([api('/api/health'),api('/api/config')]);
  $('health').textContent=h.graphkit_exists?'OpenGridFM connected':'OpenGridFM unavailable';
  $('backend-source').textContent=h.graphkit_exists?'Local GraphKit source detected':'Set GENCO_ROOT to the OpenGridFM directory';
  if(!c.configs.length){
   $('setup-warning').hidden=false;
   $('setup-warning').textContent='GraphKit configurations were not found. Set GENCO_ROOT to an OpenGridFM directory containing gridfm-graphkit/examples/config, then restart the server.';
   $('health').textContent='Setup required';
   $('tour-results').disabled=true;
   syncControls();
   return;
  }
  $('config').replaceChildren(...c.configs.map(x=>new Option(x,x)));
  $('config').value=c.configs.includes('hello_world_train_case14.yaml')?'hello_world_train_case14.yaml':c.configs[0];
  $('case').replaceChildren(...c.cases.map(x=>new Option(x,x)));
  $('compare').replaceChildren(new Option('No comparison',''),...c.cases.map(x=>new Option(x,x)));
  $('provider').value=c.provider.provider;$('model').value=c.provider.model;
  setupReady=true;
  await Promise.all([loadParameters(),refreshJobs(),...(c.cases.length?[loadCase()]:[])]);
  selectGroup('projection');
  if(!c.cases.length){
   $('setup-warning').hidden=false;
   $('setup-warning').textContent='No saved DataKit AC cases were found under GENCO_ROOT/test/results. Architecture and training configuration remain available; power-flow comparisons require those result files.';
   $('tour-results').disabled=true;
   $('case').disabled=true;$('benchmark').disabled=true;
   document.querySelectorAll('[data-engine]').forEach(button=>button.disabled=true);
   $('source').textContent='No saved power-flow case is available.';
   $('report-type').value='run';
  }
  syncControls();
  const available=await api('/api/models');
  $('model-list').replaceChildren(...available.models.map(x=>new Option(x,x)));
  $('model-status').textContent=available.available?'Ollama online · '+available.models.length+' local models':available.message;
  if(available.models.length&&!available.models.includes($('model').value))$('model-status').textContent+='; selected model is not installed, choose from the list';
 }catch(e){
  $('setup-warning').hidden=false;
  $('setup-warning').textContent='Initialization failed: '+(e.message||String(e))+'. Check GENCO_ROOT and reload after fixing the server.';
  problem(e);
 }
}
let polling=false;setInterval(async()=>{if(polling||document.hidden)return;polling=true;try{await refreshJobs();await Promise.all([refreshSelectedJob(),pollBenchmarks(),refreshPlan()])}catch{}finally{polling=false}},3000);
let savedDraft;try{savedDraft=JSON.parse(localStorage.getItem('genco-draft-v1')||'null')}catch{}
syncControls();
init().then(async()=>{if(!description)return;$('report-case').replaceChildren(...[...$('case').options].map(o=>new Option(o.text,o.value)));reportControls();if(savedDraft&&[...$('config').options].some(o=>o.value===savedDraft.config)){try{$('config').value=savedDraft.config;await loadParameters();$('data-path').value=savedDraft.data_path||'';applyChanges(savedDraft.values);await refreshPreview();$('message').value=savedDraft.message||'';$('route-mode').value=savedDraft.route||'';if(savedDraft.plan_id)showPlan(await api('/api/plans/'+savedDraft.plan_id));notify('Local parameter draft restored; automatic execution does not resume by itself.')}catch(e){notify('Draft partially restored: '+e.message,true)}}syncControls()});
