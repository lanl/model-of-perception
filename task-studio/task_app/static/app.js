const $ = id => document.getElementById(id);
let catalog, busy = false, revision = 0, listedCombinations = [];
const excludedCombinations = new Set();
const combinationVisualization = new CombinationViz.View({
 canvas:$('combinationCanvas'), inspector:$('combinationInspect'), info:$('combinationTrace'),
 toggle:$('toggleInspectedCombination'), onToggle:toggleCombination,
});
function toggleCombination(id){
 if(excludedCombinations.has(id))excludedCombinations.delete(id);else excludedCombinations.add(id);
 showCombinations({combination_options:listedCombinations.map(item=>({...item,included:!excludedCombinations.has(item.id)})),
  count:listedCombinations.filter(item=>!excludedCombinations.has(item.id)).length,compatible_count:listedCombinations.length});
 check();
}
function setCombinationView(view){
 const visual=view==='visualization';
 $('combinationVisualization').hidden=!visual;$('combinationList').hidden=visual;
 $('combinationVisButton').setAttribute('aria-pressed',String(visual));
 $('combinationListButton').setAttribute('aria-pressed',String(!visual));
}
$('combinationVisButton').addEventListener('click',()=>setCombinationView('visualization'));
$('combinationListButton').addEventListener('click',()=>setCombinationView('list'));
const headings = {datasets:'Dataset',colormaps:'Colormap',operation:'Operation',selection:'Selection',aggregation:'Aggregation',action:'Action'};
const defaults = {datasets:['terrain_100'],colormaps:['gray'],operation:['identity'],selection:['point','regions','field'],aggregation:['identity','mean','minimum','maximum'],action:['retrieve','compare','find']};
function status(message,error=false){$('status').textContent=message;$('status').className=error?'error':'';}
async function api(path,data){const response=await fetch(path,data?{method:'POST',headers:{'Content-Type':'application/json','X-CSRF-Token':catalog?.csrf_token||''},body:JSON.stringify(data)}:{});const result=await response.json();if(!response.ok)throw Error(result.error||response.statusText);return result;}
function selections(){const result={};for(const key of Object.keys(headings))result[key]=[...document.querySelectorAll(`input[data-column="${key}"]:checked`)].map(x=>x.value);result.excluded_combinations=[...excludedCombinations].sort();return result;}
function renderColumns(previous){$('columns').replaceChildren();for(const [key,title] of Object.entries(headings)){
 const column=document.createElement('div');column.className='column';const header=document.createElement('div');header.className='column-head';const h=document.createElement('h3');h.textContent=title;header.append(h);
 const toggle=document.createElement('button');toggle.className='mini';toggle.textContent='all / none';toggle.addEventListener('click',()=>{const boxes=[...column.querySelectorAll('input')];const checked=boxes.some(x=>!x.checked);boxes.forEach(x=>x.checked=checked);check();});header.append(toggle);column.append(header);
 const list=document.createElement('div');list.className='options';const options=key==='datasets'?catalog.datasets:key==='colormaps'?catalog.colormaps:Object.entries(catalog.options[key]).map(([id,name])=>({id,name}));
 for(const option of options){const label=document.createElement('label');label.className='option';const input=document.createElement('input');input.type='checkbox';input.value=option.id;input.dataset.column=key;input.checked=previous?previous[key].includes(option.id):!defaults[key]||defaults[key].includes(option.id);input.addEventListener('change',check);const text=document.createElement('span');text.textContent=option.name;
 if(option.colors){const swatch=document.createElement('span');swatch.className='swatch';const colors=option.colors.filter((_,i)=>i%Math.max(1,Math.floor(option.colors.length/20))===0).map(c=>`rgb(${c.slice(0,3).map(v=>Math.round(v*255)).join(',')})`);swatch.style.background=`linear-gradient(to right,${colors.join(',')})`;text.append(swatch);}
 if(option.shape)label.title=`${option.shape.join(' × ')} cells · axes ${option.extent.join(' to ')}${option.source?'\n'+option.source.resampling:''}`;label.append(input,text);list.append(label);}column.append(list);$('columns').append(column);}}
async function check(){const current=++revision;const selected=selections();$('generate').disabled=true;
 if(Object.keys(headings).some(key=>!selected[key].length)){$('compatibility').textContent='Select at least one option in each column.';$('parameterSummary').hidden=true;showCombinations({combination_options:[],count:0,compatible_count:0});$('parameterRows').replaceChildren();$('parameterNote').textContent='Select options in every column to calculate the parameter space.';return;}
 try{const result=await api('/api/compatibility',selected);if(current!==revision)return;$('compatibility').textContent=`${result.count.toLocaleString()} stage combinations included · ${(result.compatible_count-result.count).toLocaleString()} excluded`; showCombinations(result); showParameterSpace(result.parameter_space); $('generate').disabled=busy||result.count===0;
 // Keep options toggleable; explain checked choices that currently have no matching path.
 for(const key of Object.keys(catalog.options)){for(const input of document.querySelectorAll(`input[data-column="${key}"]`)){const viable=result.combinations.some(c=>c[key]===input.value);input.parentElement.title=input.checked&&!viable?'No included compatible path with the current choices. Change the columns or include a combination below.':'';}}
 }catch(error){status(error.message,true);}}
function showCombinations(result){
 listedCombinations=result.combination_options;
 $('combinationCount').textContent=`· ${result.count} included / ${result.compatible_count} compatible`;
 $('combinationRows').replaceChildren();
 $('includeCombinations').disabled=!listedCombinations.length;
 $('excludeCombinations').disabled=!listedCombinations.length;
 combinationVisualization.render(listedCombinations,catalog.options);
 for(const item of listedCombinations){
  const row=document.createElement('tr');row.className=item.included?'':'excluded-combination';
  const td=document.createElement('td');const label=document.createElement('label');label.className='combination-toggle';
  const checkbox=document.createElement('input');checkbox.type='checkbox';checkbox.checked=item.included;
  checkbox.setAttribute('aria-label','Include '+Object.entries(item.stages).map(([key,value])=>catalog.options[key][value]).join(' → '));
  const text=document.createElement('span');text.textContent=item.included?'Included':'Excluded';
  checkbox.addEventListener('change',()=>{if(checkbox.checked)excludedCombinations.delete(item.id);else excludedCombinations.add(item.id);text.textContent=checkbox.checked?'Included':'Excluded';row.className=checkbox.checked?'':'excluded-combination';check();});
  label.append(checkbox,text);td.append(label);row.append(td);
  for(const [key,value] of Object.entries(item.stages))cell(row,catalog.options[key][value]);
  $('combinationRows').append(row);
 }
}
$('includeCombinations').addEventListener('click',()=>{listedCombinations.forEach(item=>excludedCombinations.delete(item.id));check();});
$('excludeCombinations').addEventListener('click',()=>{listedCombinations.forEach(item=>excludedCombinations.add(item.id));check();});
function showParameterSpace(space){
 $('parameterSummaryRows').replaceChildren();$('parameterSummary').hidden=!space.totals.length;
 for(const total of space.totals){
  const row=document.createElement('li'),label=document.createElement('strong');
  const dimensions=total.continuous_dimensions,count=BigInt(total.discrete_settings).toLocaleString();
  label.textContent=dimensions?`With ${dimensions} continuous parameter${dimensions===1?'':'s'}: `:'Only discrete parameters: ';
  row.append(label,document.createTextNode(dimensions
   ?`up to ${count} combinations of discrete choices, with ${dimensions===1?'the continuous value':'the continuous values'} free to vary within each.`
   :`up to ${count} task variants.`));
  $('parameterSummaryRows').append(row);
 }
 $('parameterNote').textContent=space.note;$('parameterRows').replaceChildren();for(const item of space.rows){const row=document.createElement('tr');cell(row,item.dataset);cell(row,Object.entries(item.stages).map(([k,v])=>catalog.options[k][v]).join(' → '));cell(row,item.continuous_dimensions);cell(row,BigInt(item.discrete_settings).toLocaleString());cell(row,item.factors.map(f=>f.kind==='discrete'?`${f.name}: ${BigInt(f.choices).toLocaleString()} choices`:`${f.name}: ${f.dimensions} continuous parameter${f.dimensions===1?'':'s'}`).join('; ')||'No free parameters');$('parameterRows').append(row);}}
function cell(row,value,cls){const td=document.createElement('td');if(cls)td.className=cls;td.textContent=value;row.append(td);return td;}
function structured(row,value,collapse=false){const td=document.createElement('td');const pre=document.createElement('pre');pre.textContent=JSON.stringify(value,null,2);if(collapse){const detail=document.createElement('details');const summary=document.createElement('summary');summary.textContent='View all accepted answers';detail.append(summary,pre);td.append(detail);}else td.append(pre);row.append(td);}
function showResults(id,result){$('results').hidden=false;$('rows').replaceChildren();$('downloads').replaceChildren();$('resultCount').textContent=`${result.records.length} / ${result.manifest.requested_count} tasks`;
 for(const [file,label] of [['everything.zip','↓ Everything ZIP'],['tasks.csv','↓ Table CSV'],['images.zip','↓ Images ZIP'],['tasks.md','↓ Illustrated Markdown']]){const a=document.createElement('a');a.href=`/downloads/${id}/${file}`;a.className='download';a.download=file;a.textContent=label;$('downloads').append(a);}
 for(const record of result.records){const tr=document.createElement('tr');for(const key of ['task_id','dataset','colormap','operation','selection','aggregation','action'])cell(tr,catalog.options[key]?.[record[key]]||record[key]);const td=document.createElement('td');const a=document.createElement('a');a.href=`/downloads/${id}/${record.image_ref}`;a.target='_blank';a.rel='noopener';const image=document.createElement('img');image.src=a.href;image.alt=`Visualization for ${record.task_id}`;image.loading='lazy';a.append(image);td.append(a);tr.append(td);cell(tr,record.parameter_summary.join('\n'),'sampled-parameters');cell(tr,record.prompt,'prompt');structured(tr,record.ground_truth,record.answer_spec.type==='point'||Boolean(record.ground_truth?.accepted_xy));$('rows').append(tr);}
 $('failures').hidden=!result.manifest.failures.length;$('failures').querySelector('pre').textContent=JSON.stringify(result.manifest.failures,null,2);$('results').scrollIntoView({behavior:'smooth',block:'start'});}
$('generate').addEventListener('click',async()=>{if(busy)return;const count=Number($('count').value),seed=Number($('seed').value);if(!Number.isInteger(count)||count<1||count>(catalog?.limits?.max_tasks||50)||!Number.isSafeInteger(seed)||seed<0){status(`Enter a task count from 1 to ${catalog?.limits?.max_tasks||50} and a nonnegative integer seed.`,true);return;}
 busy=true;$('generate').disabled=true;$('generate').textContent='Generating…';try{const {id}=await api('/api/generate',{...selections(),count,seed});while(true){const result=await api(`/api/jobs/${id}`);status(result.status==='queued'?'Your batch is queued. Generation will start when the current batch finishes.':`Generating ${result.completed} / ${result.total} tasks…`);if(result.status==='error')throw Error(result.error);if(result.status==='complete'){showResults(id,result);status(`Ready: ${result.records.length} tasks. ${result.manifest.failures.length?`${result.manifest.failures.length} slots could not produce a valid, distinct task; details are below.`:'All tasks are valid and distinct.'}`);break;}await new Promise(resolve=>setTimeout(resolve,1000));}}catch(error){status(error.message,true);}finally{busy=false;$('generate').textContent='Generate tasks ↗';check();}});
$('selectAll').addEventListener('click',()=>{document.querySelectorAll('input[data-column]').forEach(input=>input.checked=true);check();});
for(const [id,kind] of [['datasetUpload','dataset'],['colormapUpload','colormap']]){$(id).addEventListener('change',async event=>{const file=event.target.files[0];if(!file)return;try{if(file.size>2_500_000)throw Error('Upload limit: 2.5 MB.');const bytes=new Uint8Array(await file.arrayBuffer());let binary='';for(let i=0;i<bytes.length;i+=8192)binary+=String.fromCharCode(...bytes.subarray(i,i+8192));const previous=selections();const result=await api('/api/upload',{kind,filename:file.name,content:btoa(binary),extent:[Number($('axisMin').value),Number($('axisMax').value)]});catalog={...catalog,...result};previous[kind==='dataset'?'datasets':'colormaps'].push(result.id);renderColumns(previous);check();status(`${file.name} uploaded and selected.`);}catch(error){status(error.message,true);}finally{event.target.value='';}});}
api('/api/catalog').then(result=>{catalog=result;$('count').max=result.limits.max_tasks;$('count').value=Math.min(Number($('count').value),result.limits.max_tasks);$('retention').textContent=`Your uploads and downloads belong to this browser session. Download bundles within ${result.limits.download_minutes} minutes. The latest five batches are kept; restarting the service clears sessions and downloads.`;const warnings=result.dataset_warnings||[];$('datasetWarnings').hidden=!warnings.length;$('datasetWarnings').textContent=warnings.length?'Some prepared datasets could not be loaded: '+warnings.join('; '):'';renderColumns();check();}).catch(error=>status(error.message,true));
