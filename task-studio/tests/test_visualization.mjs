// Run with Node: node tests/test_visualization.mjs
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
const source = readFileSync(new URL('../task_app/static/combinations-vis.js', import.meta.url), 'utf8');
const context = vm.createContext({});
vm.runInContext(source, context);
const {buildLayout, colorFor} = context.CombinationViz;
const options = {
 operation: {identity:'Original scalar field',gradient:'Gradient magnitude'},
 selection: {point:'Point',regions:'Two rectangles',field:'Whole field'},
 aggregation: {identity:'No reduction',mean:'Mean',maximum:'Maximum'},
 action: {retrieve:'Retrieve',compare:'Compare',find:'Find'},
};
const combinations = [
 {id:'identity|point|identity|retrieve',included:true,stages:{operation:'identity',selection:'point',aggregation:'identity',action:'retrieve'}},
 {id:'identity|regions|mean|compare',included:true,stages:{operation:'identity',selection:'regions',aggregation:'mean',action:'compare'}},
 {id:'identity|field|maximum|find',included:false,stages:{operation:'identity',selection:'field',aggregation:'maximum',action:'find'}},
];
const layout = buildLayout(combinations, options);
assert.equal(layout.paths.length,3);
for(const path of layout.paths){
 assert.equal(path.points.length,4);
 assert.equal((path.d.match(/ C /g)||[]).length,3);
 assert(!/NaN|undefined/.test(path.d));
 path.points.forEach(([x,y],stage)=>{
  const node=layout.nodes.find(n=>n.key===layout.columns[stage].key&&n.value===path.stages[n.key]);
  assert.equal(x,node.x);
  assert(y>=node.y+11&&y<=node.y+29);
  assert(x>=0&&x<=layout.width&&y>=0&&y<=layout.height);
 });
}
const reordered=buildLayout([...combinations].reverse(),options);
const excluded=buildLayout(combinations.map(item=>({...item,included:false})),options);
for(const path of layout.paths){
 assert.equal(path.d,reordered.paths.find(p=>p.id===path.id).d);
 assert.equal(path.d,excluded.paths.find(p=>p.id===path.id).d);
 assert.equal(path.color,colorFor(path.id));
}
assert.equal(buildLayout([],options).paths.length,0);
assert.equal(buildLayout([combinations[0]],options).paths.length,1);
new vm.Script(readFileSync(new URL('../task_app/static/app.js',import.meta.url),'utf8'));
console.log('Visualization checks passed: all four stages, spline geometry, exclusions, stable layout/colors, empty/single states, frontend syntax.');

// Exercise the actual SVG component's event/state handling without a browser.
class Element {
 constructor(){this.children=[];this.attributes={};this.dataset={};this.style={};this.handlers={};this.value='';
  this.classList={toggle:(name,on)=>{const set=new Set((this.attributes.class||'').split(' ').filter(Boolean));if(on)set.add(name);else set.delete(name);this.attributes.class=[...set].join(' ');},contains:name=>(this.attributes.class||'').split(' ').includes(name)};
 }
 setAttribute(key,value){this.attributes[key]=String(value);if(key.startsWith('data-'))this.dataset[key.slice(5).replace(/-([a-z])/g,(_,c)=>c.toUpperCase())]=String(value);}
 append(...children){this.children.push(...children);}
 replaceChildren(...children){this.children=children;}
 addEventListener(event,handler){(this.handlers[event]||=[]).push(handler);}
 emit(event,props={}){for(const handler of this.handlers[event]||[])handler(props);}
 querySelectorAll(selector){return this.children.flatMap(child=>[...(child.classList?.contains(selector.slice(1))?[child]:[]),...(child.querySelectorAll?.(selector)||[])]);}
 focus(){context.document.activeElement=this;this.emit('focus');}
}
context.document={activeElement:null,createElement:()=>new Element(),createElementNS:()=>new Element()};
const canvas=new Element(), inspector=new Element(),info=new Element(),toggle=new Element();
let current=combinations.map(item=>({...item}));
const view=new context.CombinationViz.View({canvas,inspector,info,toggle,onToggle:id=>{
 current=current.map(item=>item.id===id?{...item,included:!item.included}:item);
 view.render(current,options);
}});
view.render(current,options);
const first=current[0].id;
assert.equal(view.groups.get(first).attributes['aria-pressed'],'true');
view.groups.get(first).emit('pointerenter');
assert(info.textContent.includes('Original scalar field'));
assert(view.groups.get(first).classList.contains('is-highlighted'));
view.groups.get(first).emit('click');
assert.equal(view.groups.get(first).attributes['aria-pressed'],'false');
assert.equal(inspector.value,first);
assert.equal(toggle.textContent,'Include selected');
toggle.emit('click');
assert.equal(view.groups.get(first).attributes['aria-pressed'],'true');
let prevented=false;
view.groups.get(first).emit('keydown',{key:' ',preventDefault(){prevented=true;}});
assert(prevented);
assert.equal(view.groups.get(first).attributes['aria-pressed'],'false');
inspector.value=current[1].id;inspector.emit('change');
assert.equal(view.selected,current[1].id);
assert.equal(toggle.textContent,'Exclude selected');
view.render([],options);
assert.equal(inspector.disabled,true);assert.equal(toggle.disabled,true);assert.equal(view.selected,'');
view.render(current,options);
assert.equal(view.groups.get(first).attributes['aria-pressed'],'false');
console.log('Component interaction checks passed: curve toggle, keyboard, hover, inspector, empty state, and exclusion preservation.');
