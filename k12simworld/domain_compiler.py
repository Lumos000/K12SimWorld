"""Compile trusted domain-solver traces into self-contained Canvas scenes."""

from __future__ import annotations

import copy
import hashlib
import html
import json
from typing import Any, Dict, List, Mapping

from .domain_solvers import DOMAIN_ENGINES, DomainSimulationError, simulate_domain
from .models import EduWorldSpec


SOLVER_VERSION = "k12-domain-solvers-v1.1"


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _script_json(value: Any) -> str:
    # Prevent data strings from terminating the containing script element.
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def _render_trace(trace: Mapping[str, Any], maximum_frames: int = 600) -> Dict[str, Any]:
    result = copy.deepcopy(dict(trace))
    series = result.get("time_series")
    if isinstance(series, list) and len(series) > maximum_frames:
        indexes = {
            round(index * (len(series) - 1) / (maximum_frames - 1))
            for index in range(maximum_frames)
        }
        result["time_series"] = [series[index] for index in sorted(indexes)]
    return result


def _shell(
    title: str,
    engine: str,
    spec: Mapping[str, Any],
    trace: Mapping[str, Any],
    world_object_ids: List[str],
    drawing_script: str,
) -> str:
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
html,body{{margin:0;width:100%;height:100%;overflow:hidden;background:#f8fafc;font-family:Arial,sans-serif}}
#stage{{width:100%;height:100%;display:block}} .hidden{{display:none}}
</style></head><body>
<canvas id="stage" width="512" height="512" aria-label="{html.escape(title)}"></canvas>
<div class="hidden" data-engine="{html.escape(engine)}" data-world-objects="{html.escape(' '.join(world_object_ids))}"></div>
<script>
'use strict';
const SPEC={_script_json(spec)};
const TRACE={_script_json(_render_trace(trace))};
const WORLD_OBJECT_IDS={_script_json(world_object_ids)};
const canvas=document.getElementById('stage'); const ctx=canvas.getContext('2d');
const W=canvas.width,H=canvas.height, started=performance.now();
function clear(){{ctx.fillStyle='#f8fafc';ctx.fillRect(0,0,W,H);}}
function text(value,x,y,color='#0f172a',size=13,align='left'){{ctx.fillStyle=color;ctx.font=size+'px Arial';ctx.textAlign=align;ctx.fillText(String(value),x,y);}}
function arrow(x1,y1,x2,y2,color='#334155'){{ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();const a=Math.atan2(y2-y1,x2-x1);ctx.beginPath();ctx.moveTo(x2,y2);ctx.lineTo(x2-8*Math.cos(a-.45),y2-8*Math.sin(a-.45));ctx.lineTo(x2-8*Math.cos(a+.45),y2-8*Math.sin(a+.45));ctx.closePath();ctx.fill();}}
{drawing_script}
requestAnimationFrame(draw);
</script></body></html>"""


def _equation_html(spec: Mapping[str, Any], trace: Mapping[str, Any], ids: List[str]) -> str:
    script = r"""
const b=TRACE.bounds, pad=52;
const sx=x=>pad+(x-b.x_min)/(b.x_max-b.x_min)*(W-2*pad);
const sy=y=>H-pad-(y-b.y_min)/(b.y_max-b.y_min)*(H-2*pad);
function draw(){
  clear(); const frames=TRACE.time_series; const duration=Math.max(TRACE.playback_duration||8,.001);
  const t=((performance.now()-started)/1000)%duration;
  let idx=Math.min(frames.length-1,Math.floor(t/duration*frames.length));
  (TRACE.field_regions||[]).forEach(r=>{const q=r.bounds;ctx.globalAlpha=.28;ctx.fillStyle=r.color;ctx.fillRect(sx(q.x_min),sy(q.y_max),sx(q.x_max)-sx(q.x_min),sy(q.y_min)-sy(q.y_max));ctx.globalAlpha=1;text(r.label,(sx(q.x_min)+sx(q.x_max))/2,sy(q.y_max)+15,'#475569',11,'center');});
  ctx.strokeStyle='#cbd5e1';ctx.lineWidth=1;
  for(let i=0;i<=10;i++){let x=pad+i*(W-2*pad)/10;ctx.beginPath();ctx.moveTo(x,pad);ctx.lineTo(x,H-pad);ctx.stroke();let y=pad+i*(H-2*pad)/10;ctx.beginPath();ctx.moveTo(pad,y);ctx.lineTo(W-pad,y);ctx.stroke();}
  ctx.strokeStyle='#475569';ctx.strokeRect(pad,pad,W-2*pad,H-2*pad);
  text('带电粒子：q(E + v × B)',16,24,'#0f172a',17);
  const E=TRACE.fields.electric,B=TRACE.fields.magnetic;
  text(`E=(${E[0]}, ${E[1]})`,16,45,'#475569'); text(`Bz=${B[2]}`,180,45,'#475569');
  TRACE.particles.forEach(p=>{
    ctx.strokeStyle=p.color;ctx.lineWidth=2;ctx.beginPath();
    for(let j=0;j<=idx;j++){const q=frames[j].objects[p.id].position;const x=sx(q[0]),y=sy(q[1]);if(j===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);}ctx.stroke();
    const state=frames[idx].objects[p.id],pos=state.position,x=sx(pos[0]),y=sy(pos[1]);
    ctx.fillStyle=p.color;ctx.beginPath();ctx.arc(x,y,7,0,Math.PI*2);ctx.fill();
    arrow(x,y,x+state.velocity[0]*18,y-state.velocity[1]*18,'#0f172a');
    text(`${p.label}  t=${frames[idx].t}s  |v|=${state.speed}`,W/2,H-18,p.color,13,'center');
  });
  requestAnimationFrame(draw);
}
"""
    return _shell("Charged-particle equation simulation", "equation-solver", spec, trace, ids, script)


def _ode_html(spec: Mapping[str, Any], trace: Mapping[str, Any], ids: List[str]) -> str:
    script = r"""
const frames=TRACE.time_series, channels=TRACE.plot_channels.slice(0,6), bindings=(TRACE.visual_bindings||[]).slice(0,4);
const palette=['#2563eb','#dc2626','#16a34a','#9333ea','#ea580c','#0891b2'];
const pad={left:62,right:18,top:bindings.length?190:70,bottom:72};
const x0=pad.left,x1=W-pad.right,y0=pad.top,y1=H-pad.bottom;
const rangeByChannel=TRACE.summary.channel_ranges;
function value(frame,channel){return Object.prototype.hasOwnProperty.call(frame.state,channel)?frame.state[channel]:frame.observables[channel];}
function sx(t){return x0+t/Math.max(TRACE.duration,1e-12)*(x1-x0);}
function sy(v,channel){const r=rangeByChannel[channel],span=Math.max(r.maximum-r.minimum,1e-12);return y1-(v-r.minimum)/span*(y1-y0);}
function drawBinding(binding,k,frame){
  const cardW=(W-24)/Math.max(bindings.length,1),cx=12+cardW*(k+.5),cy=105;
  const raw=value(frame,binding.channel),r=rangeByChannel[binding.channel];
  const lo=binding.minimum===undefined?r.minimum:binding.minimum,hi=binding.maximum===undefined?r.maximum:binding.maximum;
  const u=Math.max(0,Math.min(1,(raw-lo)/Math.max(hi-lo,1e-12)));
  ctx.strokeStyle='#cbd5e1';ctx.fillStyle='#ffffff';ctx.lineWidth=1;ctx.fillRect(cx-cardW/2+4,48,cardW-8,116);ctx.strokeRect(cx-cardW/2+4,48,cardW-8,116);
  if(binding.type==='slider'){
    ctx.strokeStyle='#64748b';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(cx-cardW*.32,105);ctx.lineTo(cx+cardW*.32,105);ctx.stroke();
    ctx.fillStyle='#2563eb';ctx.fillRect(cx-cardW*.32+u*cardW*.64-10,88,20,34);
  }else if(binding.type==='rotor'){
    ctx.strokeStyle='#2563eb';ctx.lineWidth=3;ctx.beginPath();ctx.arc(cx,105,27,0,Math.PI*2);ctx.stroke();
    const angle=raw;arrow(cx,105,cx+24*Math.cos(angle),cy-24*Math.sin(angle),'#dc2626');
  }else if(binding.type==='lamp'){
    ctx.fillStyle=`rgba(250,204,21,${.12+.88*u})`;ctx.strokeStyle='#ca8a04';ctx.lineWidth=2;ctx.beginPath();ctx.arc(cx,102,27,0,Math.PI*2);ctx.fill();ctx.stroke();
    ctx.beginPath();ctx.moveTo(cx-15,87);ctx.lineTo(cx+15,117);ctx.moveTo(cx+15,87);ctx.lineTo(cx-15,117);ctx.stroke();
  }else{
    ctx.strokeStyle='#64748b';ctx.lineWidth=3;ctx.beginPath();ctx.arc(cx,112,29,Math.PI,2*Math.PI);ctx.stroke();
    const angle=Math.PI*(1+u);arrow(cx,112,cx+25*Math.cos(angle),112+25*Math.sin(angle),'#dc2626');
  }
  text(binding.label,cx,65,'#0f172a',11,'center');text(`${binding.channel}=${Number(raw).toPrecision(4)}`,cx,151,'#475569',10,'center');
}
function draw(){
  clear(); const duration=Math.max(TRACE.playback_duration||8,.001);
  const elapsed=((performance.now()-started)/1000)%duration;
  const idx=Math.min(frames.length-1,Math.floor(elapsed/duration*frames.length));
  text('方程系统：受限表达式 + RK4',16,25,'#0f172a',17);
  text(`物理时刻 t=${frames[idx].t}`,W-16,25,'#475569',13,'right');
  bindings.forEach((binding,k)=>drawBinding(binding,k,frames[idx]));
  ctx.strokeStyle='#cbd5e1';ctx.lineWidth=1;
  for(let i=0;i<=5;i++){const y=y0+i*(y1-y0)/5;ctx.beginPath();ctx.moveTo(x0,y);ctx.lineTo(x1,y);ctx.stroke();}
  for(let i=0;i<=5;i++){const x=x0+i*(x1-x0)/5;ctx.beginPath();ctx.moveTo(x,y0);ctx.lineTo(x,y1);ctx.stroke();text((TRACE.duration*i/5).toPrecision(3),x,y1+18,'#64748b',10,'center');}
  ctx.strokeStyle='#475569';ctx.strokeRect(x0,y0,x1-x0,y1-y0);
  channels.forEach((channel,k)=>{
    ctx.strokeStyle=palette[k%palette.length];ctx.lineWidth=2;ctx.beginPath();
    for(let j=0;j<=idx;j++){const x=sx(frames[j].t),y=sy(value(frames[j],channel),channel);if(j===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);}ctx.stroke();
    const r=rangeByChannel[channel],current=value(frames[idx],channel);
    const col=k%3,row=Math.floor(k/3),lx=20+col*165,ly=H-42+row*18;
    text(`${channel}=${Number(current).toPrecision(4)}  [${Number(r.minimum).toPrecision(3)}, ${Number(r.maximum).toPrecision(3)}]`,lx,ly,palette[k%palette.length],10);
  });
  const cursor=sx(frames[idx].t);ctx.strokeStyle='#0f172a';ctx.setLineDash([5,4]);ctx.beginPath();ctx.moveTo(cursor,y0);ctx.lineTo(cursor,y1);ctx.stroke();ctx.setLineDash([]);
  requestAnimationFrame(draw);
}
"""
    return _shell("Deterministic equation-system simulation", "equation-solver", spec, trace, ids, script)


def _circuit_html(spec: Mapping[str, Any], trace: Mapping[str, Any], ids: List[str]) -> str:
    script = r"""
const nodes=TRACE.nodes.map((n,i)=>({...n}));
if(nodes.every(n=>n.x===0&&n.y===0)){nodes.forEach((n,i)=>{const a=2*Math.PI*i/nodes.length-Math.PI/2;n.x=Math.cos(a);n.y=Math.sin(a);});}
const xs=nodes.map(n=>n.x),ys=nodes.map(n=>n.y),xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys),pad=72;
const sx=x=>pad+(x-xmin)/(Math.max(xmax-xmin,1e-6))*(W-2*pad), sy=y=>H-pad-(y-ymin)/(Math.max(ymax-ymin,1e-6))*(H-2*pad);
const byNode=Object.fromEntries(nodes.map(n=>[n.id,n]));
function symbol(c,state,x,y,angle){ctx.save();ctx.translate(x,y);ctx.rotate(angle);ctx.fillStyle='#f8fafc';ctx.strokeStyle='#334155';ctx.lineWidth=2;
  if(c.type==='lamp'){ctx.fillStyle=`rgba(250,204,21,${0.15+0.85*(state.brightness||0)})`;ctx.beginPath();ctx.arc(0,0,16,0,Math.PI*2);ctx.fill();ctx.stroke();ctx.beginPath();ctx.moveTo(-10,-10);ctx.lineTo(10,10);ctx.moveTo(10,-10);ctx.lineTo(-10,10);ctx.stroke();}
  else if(c.type==='ammeter'||c.type==='voltmeter'){ctx.beginPath();ctx.arc(0,0,16,0,Math.PI*2);ctx.fill();ctx.stroke();text(c.type==='ammeter'?'A':'V',0,5,'#0f172a',15,'center');}
  else if(c.type==='switch'){ctx.beginPath();ctx.moveTo(-17,0);ctx.lineTo(17,state.closed?0:-13);ctx.stroke();ctx.beginPath();ctx.arc(-17,0,3,0,Math.PI*2);ctx.arc(17,0,3,0,Math.PI*2);ctx.fillStyle='#334155';ctx.fill();}
  else if(c.type==='voltage_source'){ctx.beginPath();ctx.moveTo(-5,-15);ctx.lineTo(-5,15);ctx.moveTo(6,-9);ctx.lineTo(6,9);ctx.stroke();}
  else{ctx.strokeRect(-18,-8,36,16);}ctx.restore();}
function draw(){clear();const frames=TRACE.time_series,duration=Math.max(TRACE.duration,.001),t=((performance.now()-started)/1000)%duration,idx=Math.min(frames.length-1,Math.floor(t/duration*frames.length)),frame=frames[idx];
  text('直流电路：KCL / KVL / Ohm',16,25,'#0f172a',17);text(`t=${frame.t}s`,W-16,25,'#475569',13,'right');
  TRACE.components.forEach(c=>{const a=byNode[c.node_a],b=byNode[c.node_b],x1=sx(a.x),y1=sy(a.y),x2=sx(b.x),y2=sy(b.y),mx=(x1+x2)/2,my=(y1+y2)/2,state=frame.components[c.id]||{};ctx.strokeStyle='#64748b';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();symbol(c,state,mx,my,Math.atan2(y2-y1,x2-x1));
    let value='';if(c.type==='ammeter'||c.type==='voltmeter')value=`${state.reading} ${state.reading_unit}`;else if(c.type==='lamp')value=`${state.power} W  ${(100*(state.brightness||0)).toFixed(0)}%`;else value=`I=${state.current} A`;text(c.label,mx,my-23,'#0f172a',12,'center');text(value,mx,my+31,'#475569',11,'center');});
  nodes.forEach(n=>{const x=sx(n.x),y=sy(n.y);ctx.fillStyle=n.id===TRACE.ground?'#0f172a':'#2563eb';ctx.beginPath();ctx.arc(x,y,5,0,Math.PI*2);ctx.fill();text(`${n.label}: ${frame.node_voltages[n.id]} V`,x,y-10,'#334155',11,'center');});requestAnimationFrame(draw);}
"""
    return _shell("Deterministic DC circuit simulation", "circuit-solver", spec, trace, ids, script)


def _optics_html(spec: Mapping[str, Any], trace: Mapping[str, Any], ids: List[str]) -> str:
    script = r"""
const all=[];TRACE.elements.forEach(e=>all.push(e.p1,e.p2));TRACE.paths.forEach(p=>p.points.forEach(q=>all.push(q)));
const xs=all.map(q=>q[0]),ys=all.map(q=>q[1]),xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys),pad=50;
const sx=x=>pad+(x-xmin)/Math.max(xmax-xmin,1e-6)*(W-2*pad),sy=y=>H-pad-(y-ymin)/Math.max(ymax-ymin,1e-6)*(H-2*pad);
function drawElement(e){const a=e.p1,b=e.p2;ctx.save();ctx.lineWidth=e.type==='mirror'?5:3;ctx.strokeStyle=e.type==='mirror'?'#2563eb':e.type==='thin_lens'?'#06b6d4':e.type==='screen'?'#475569':e.type==='refractive_interface'?'#8b5cf6':'#111827';ctx.setLineDash(e.type==='refractive_interface'?[7,5]:[]);ctx.beginPath();ctx.moveTo(sx(a[0]),sy(a[1]));ctx.lineTo(sx(b[0]),sy(b[1]));ctx.stroke();ctx.restore();text(e.label,(sx(a[0])+sx(b[0]))/2,(sy(a[1])+sy(b[1]))/2-8,ctx.strokeStyle,11,'center');}
function draw(){clear();text('几何光学：反射 / 折射 / 薄透镜',16,25,'#0f172a',17);TRACE.elements.forEach(drawElement);const progress=((performance.now()-started)%8000)/8000;
  TRACE.paths.forEach(path=>{const pts=path.points;let lengths=[],total=0;for(let i=1;i<pts.length;i++){const len=Math.hypot(sx(pts[i][0])-sx(pts[i-1][0]),sy(pts[i][1])-sy(pts[i-1][1]));lengths.push(len);total+=len;}let remain=total*progress;ctx.strokeStyle=path.color;ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(sx(pts[0][0]),sy(pts[0][1]));for(let i=1;i<pts.length&&remain>0;i++){const use=Math.min(remain,lengths[i-1]),ratio=use/Math.max(lengths[i-1],1e-9),x=pts[i-1][0]+(pts[i][0]-pts[i-1][0])*ratio,y=pts[i-1][1]+(pts[i][1]-pts[i-1][1])*ratio;ctx.lineTo(sx(x),sy(y));remain-=use;}ctx.stroke();ctx.fillStyle=path.color;ctx.beginPath();ctx.arc(sx(pts[0][0]),sy(pts[0][1]),5,0,Math.PI*2);ctx.fill();text(path.label,sx(pts[0][0]),sy(pts[0][1])-10,path.color,11,'center');});requestAnimationFrame(draw);}
"""
    return _shell("Deterministic geometric ray tracing", "ray-optics", spec, trace, ids, script)


def render_domain_html(
    engine: str,
    spec: Mapping[str, Any],
    trace: Mapping[str, Any],
    world_object_ids: List[str],
) -> str:
    if engine == "equation-solver":
        if trace.get("domain_model") == "ode_system":
            return _ode_html(spec, trace, world_object_ids)
        return _equation_html(spec, trace, world_object_ids)
    if engine == "circuit-solver":
        return _circuit_html(spec, trace, world_object_ids)
    if engine == "ray-optics":
        return _optics_html(spec, trace, world_object_ids)
    raise DomainSimulationError(f"unsupported domain engine {engine!r}")


def compile_domain_program(
    payload: Mapping[str, Any], world_spec: EduWorldSpec
) -> Dict[str, Any]:
    """Execute every declarative scene and replace any model HTML with trusted HTML."""
    result = copy.deepcopy(dict(payload))
    engine = str(result.get("engine") or "").strip().lower()
    if engine not in DOMAIN_ENGINES:
        return result
    scenes = result.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise DomainSimulationError("domain program requires a non-empty scenes list")
    object_ids = [str(item.get("id")) for item in world_spec.objects]
    compiled_scenes: List[Dict[str, Any]] = []
    for index, raw_scene in enumerate(scenes):
        if not isinstance(raw_scene, Mapping):
            raise DomainSimulationError(f"scenes[{index}] must be an object")
        scene = dict(raw_scene)
        simulation_spec = scene.get("simulation_spec")
        if not isinstance(simulation_spec, Mapping):
            raise DomainSimulationError(f"scenes[{index}] requires simulation_spec")
        trace = simulate_domain(engine, simulation_spec)
        scene["simulation_spec"] = dict(simulation_spec)
        scene["trace"] = trace
        scene["trace_sha256"] = canonical_sha256(trace)
        scene["solver_version"] = SOLVER_VERSION
        scene["document"] = render_domain_html(engine, simulation_spec, trace, object_ids)
        compiled_scenes.append(scene)
    result["scenes"] = compiled_scenes
    result["solver_version"] = SOLVER_VERSION
    return result


def validate_compiled_domain_scene(engine: str, scene: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    spec = scene.get("simulation_spec")
    trace = scene.get("trace")
    if not isinstance(spec, Mapping):
        return ["domain scene requires simulation_spec"]
    if not isinstance(trace, Mapping):
        return ["domain scene requires an executed trace"]
    try:
        expected_trace = simulate_domain(engine, spec)
    except DomainSimulationError as exc:
        return [f"invalid simulation_spec: {exc}"]
    expected_hash = canonical_sha256(expected_trace)
    declared_hash = str(scene.get("trace_sha256") or "")
    if declared_hash != expected_hash:
        errors.append("trace_sha256 does not match deterministic solver output")
    if canonical_sha256(trace) != expected_hash:
        errors.append("stored trace differs from deterministic solver output")
    if scene.get("solver_version") != SOLVER_VERSION:
        errors.append("domain scene solver_version is missing or outdated")
    return errors
