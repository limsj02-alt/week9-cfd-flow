let model, fields, diagnostics, quantity='speed';
const labels={speed:['Speed / U_lid','속도 크기 / 뚜껑 속도'],u:['u / U_lid','수평 속도 / 뚜껑 속도'],v:['v / U_lid','수직 속도 / 뚜껑 속도'],p:['p / (ρ U_lid²)','압력 / (밀도 × 뚜껑 속도²)']};
function predict(input){let a=input.map((x,i)=>(x-model.input_mean[i])/model.input_std[i]);for(const layer of model.layers){if(layer.type==='Linear')a=layer.weight.map((row,j)=>row.reduce((s,w,i)=>s+w*a[i],layer.bias[j]));else if(layer.type==='Tanh')a=a.map(Math.tanh);else if(layer.type==='ReLU')a=a.map(x=>Math.max(0,x));else if(layer.type==='LeakyReLU')a=a.map(x=>x>=0?x:layer.negative_slope*x);else if(layer.type==='Sigmoid')a=a.map(x=>1/(1+Math.exp(-x)));else throw Error('지원하지 않는 활성화 함수: '+layer.type);}return a.map((x,i)=>x*model.output_std[i]+model.output_mean[i]);}
const mainStops=[[68,1,84],[65,68,135],[42,120,142],[34,168,132],[122,209,81],[253,231,37]],errorStops=[[59,76,192],[141,176,254],[238,238,238],[247,168,137],[180,4,38]];
function color(x,stops){const v=Math.max(0,Math.min(1,x))*(stops.length-1),i=Math.min(stops.length-2,Math.floor(v)),t=v-i;return stops[i].map((a,k)=>Math.round(a+(stops[i+1][k]-a)*t));}
function plot(id,values,min,max,stops){const ctx=document.getElementById(id).getContext('2d'),image=ctx.createImageData(50,50),range=max-min||1;for(let j=0;j<50;j++)for(let i=0;i<50;i++){const k=((49-j)*50+i)*4,c=color((values[j*50+i]-min)/range,stops);image.data.set([...c,255],k);}ctx.putImageData(image,0,0);}
function plotParity(ref,ai,low,high){
  const ctx=document.getElementById('parity').getContext('2d'),left=105,top=35,size=500,bottom=top+size;
  const padding=(high-low)*.04||.04,min=low-padding,max=high+padding,pointX=x=>left+(x-min)/(max-min)*size,pointY=y=>bottom-(y-min)/(max-min)*size;
  ctx.clearRect(0,0,720,620);ctx.fillStyle='#fff';ctx.fillRect(0,0,720,620);ctx.font='17px Arial';ctx.textAlign='center';ctx.textBaseline='top';
  for(let i=0;i<=4;i++){const v=min+(max-min)*i/4,x=pointX(v),y=pointY(v);ctx.strokeStyle='#e5ebef';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x,top);ctx.lineTo(x,bottom);ctx.moveTo(left,y);ctx.lineTo(left+size,y);ctx.stroke();ctx.fillStyle='#587086';ctx.textAlign='center';ctx.fillText(v.toFixed(2),x,bottom+12);ctx.textAlign='right';ctx.textBaseline='middle';ctx.fillText(v.toFixed(2),left-12,y);ctx.textBaseline='top';}
  ctx.setLineDash([8,6]);ctx.strokeStyle='#7c8f9d';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(left,bottom);ctx.lineTo(left+size,top);ctx.stroke();ctx.setLineDash([]);
  ctx.fillStyle='rgba(30,105,163,0.30)';for(let i=0;i<ref.length;i++){ctx.beginPath();ctx.arc(pointX(ref[i]),pointY(ai[i]),2.6,0,2*Math.PI);ctx.fill();}
  ctx.strokeStyle='#34536b';ctx.lineWidth=1.5;ctx.strokeRect(left,top,size,size);ctx.fillStyle='#284c69';ctx.font='18px Arial';ctx.textAlign='center';ctx.fillText('CFD reference',left+size/2,bottom+48);ctx.save();ctx.translate(25,top+size/2);ctx.rotate(-Math.PI/2);ctx.textBaseline='top';ctx.fillText('Neural-network prediction',0,0);ctx.restore();ctx.font='16px Arial';ctx.textAlign='left';ctx.fillText('y = x',left+size-60,top+8);
}
function valueOf(q,v){return q==='speed'?Math.hypot(v[0],v[1]):v[{u:0,v:1,p:2}[q]];}
function render(){try{
  const re=Number(document.getElementById('re').value);if(!Number.isInteger(re)||re<300||re>1000)throw Error('Re는 300부터 1000까지의 정수여야 합니다.');
  const [steps,change,divergence]=diagnostics[re-300],ref=[],ai=[],diff=[];let vectorDiff=0,vectorNorm=0,maxAbs=0;
  for(let j=0;j<50;j++)for(let i=0;i<50;i++){
    const k=(j*50+i),off=((re-300)*2500+k)*3,truth=[fields[off],fields[off+1],fields[off+2]],pred=predict([(i+.5)/50,(j+.5)/50,re]);
    vectorDiff+=(pred[0]-truth[0])**2+(pred[1]-truth[1])**2;vectorNorm+=truth[0]**2+truth[1]**2;
    ref.push(valueOf(quantity,truth));ai.push(valueOf(quantity,pred));const delta=ai[k]-ref[k];diff.push(delta);maxAbs=Math.max(maxAbs,Math.abs(delta));
  }
  let low=Math.min(...ref,...ai),high=Math.max(...ref,...ai);if(quantity==='speed')low=0;if(high===low)high=low+1;
  const rmse=Math.sqrt(diff.reduce((s,v)=>s+v*v,0)/diff.length),relative=Math.sqrt(vectorDiff/vectorNorm)*100,absMax=Math.max(maxAbs,1e-12);
  plot('cfd',ref,low,high,mainStops);plot('ai',ai,low,high,mainStops);plot('difference',diff,-absMax,absMax,errorStops);plotParity(ref,ai,low,high);
  document.querySelectorAll('.re-label').forEach(x=>x.textContent='Re = '+re);
  document.getElementById('chosen').textContent='Re = '+re;document.getElementById('region').textContent=re>300?'EXTRAPOLATION':'TRAINING BOUNDARY';
  document.getElementById('quantity-label').textContent=labels[quantity][0];document.getElementById('quantity-description').textContent=labels[quantity][1];
  document.getElementById('scale-min').textContent=low.toFixed(3);document.getElementById('scale-max').textContent=high.toFixed(3);document.getElementById('diff-min').textContent=(-absMax).toFixed(3);document.getElementById('diff-max').textContent=absMax.toFixed(3);
  document.getElementById('rmse').textContent=rmse.toFixed(4);document.getElementById('max-error').textContent=maxAbs.toFixed(4);document.getElementById('relative-error').textContent=relative.toFixed(1)+'%';document.getElementById('convergence').textContent=change.toExponential(4);
  document.getElementById('error').textContent=`CFD 정상상태 변화율 ${change.toExponential(4)} < 2×10⁻⁵ · 발산 RMS ${divergence.toExponential(2)} < 10⁻⁸ · ${steps.toLocaleString()} steps`;
}catch(e){document.getElementById('error').textContent='비교 결과를 표시하지 못했습니다: '+e.message;}}
function syncFromNumber(){const input=document.getElementById('re-number'),value=Number(input.value);if(input.value!==''&&Number.isInteger(value)&&value>=300&&value<=1000){document.getElementById('re').value=value;input.setCustomValidity('');render();}else input.setCustomValidity('300부터 1000까지의 정수를 입력하세요.');}
async function readFile(name,format){const r=await fetch(name);if(!r.ok)throw Error(name+' 파일을 불러올 수 없습니다');return r[format]();}
Promise.all([readFile('model.json','json'),readFile('cfd-fields.bin','arrayBuffer'),readFile('cfd-diagnostics.json','json')]).then(([m,buffer,d])=>{
  model=m;fields=new Float32Array(buffer);diagnostics=d;if(fields.length!==701*2500*3||diagnostics.length!==701)throw Error('CFD 기준해 데이터의 크기가 올바르지 않습니다');
  let maxError=0;model.verification.inputs.forEach((input,i)=>predict(input).forEach((v,j)=>{maxError=Math.max(maxError,Math.abs(v-model.verification.outputs[i][j]));}));if(maxError>model.verification.absolute_tolerance)throw Error('모델 검증 오차 초과: '+maxError);
  document.getElementById('verify').textContent='PASS · '+maxError.toExponential(1);
  document.getElementById('re').addEventListener('input',()=>{document.getElementById('re-number').value=document.getElementById('re').value;document.getElementById('re-number').setCustomValidity('');render();});document.getElementById('re-number').addEventListener('input',syncFromNumber);
  document.querySelectorAll('[data-quantity]').forEach(button=>button.addEventListener('click',()=>{quantity=button.dataset.quantity;document.querySelectorAll('[data-quantity]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));render();}));render();
}).catch(e=>{document.getElementById('verify').textContent='FAILED';document.getElementById('error').textContent=e.message;});
