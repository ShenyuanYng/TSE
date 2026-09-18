import fs from 'node:fs/promises';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
import {resolvePresentationFont,finalizePresentation} from '/Users/shenyuanyang/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.61513/skills/presentations/container_tools/artifact_tool_utils.mjs';
const root='/Users/shenyuanyang/Desktop/new_example_code';
const skill='/Users/shenyuanyang/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.61513/skills/presentations';
const font=resolvePresentationFont({fontFamily:'Arial'});
const p=Presentation.create({slideSize:{width:1440,height:1000}});const s=p.slides.add();s.background.fill='#FFFFFF';
function box(text,x,y,w,h,opts={}){let a=s.shapes.add({geometry:opts.plain?'textbox':'roundRect',position:{left:x,top:y,width:w,height:h},fill:opts.plain?'none':(opts.fill||'#FFFFFF'),line:{fill:opts.plain?'none':'#111111',width:opts.plain?0:2.5},borderRadius:opts.plain?0:10});a.text=text;a.text.style={typeface:font,fontSize:opts.size||23,color:'#111111',alignment:opts.align||'center',verticalAlignment:'middle',autoFit:'none',insets:{left:8,right:8,top:3,bottom:3},bold:opts.bold||false};return a;}
function bg(x,y,w,h,color){return s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill:color,line:{fill:'none',width:0}});}
function link(a,b,from='top',to='bottom',kind='straight',dash=false){let c=s.shapes.connect(a,b,{kind,fromSide:from,toSide:to,line:{fill:'#111111',width:dash?1.6:2,style:dash?'dashed':'solid'},tail:{type:dash?'none':'triangle',width:'med',length:'med'}});c.bringToFront();return c;}
function dotted(x1,y1,x2,y2){const dx=x2-x1,dy=y2-y1,len=Math.hypot(dx,dy);s.shapes.add({geometry:'line',position:{left:(x1+x2-len)/2,top:(y1+y2)/2,width:len,height:0,rotation:Math.atan2(dy,dx)*180/Math.PI},fill:'none',line:{fill:'#111111',width:1.5,style:'dashed'}});}
box('Target-class conditioned CRNN',45,22,900,45,{plain:true,size:32,align:'left'});
bg(40,170,1360,710,'#F2F2F2'); bg(455,441,530,403,'#D9D9D9');

const out=box('Target-class frame probabilities  [B, T]',480,91,480,53,{size:24});
const head=box('Linear 256 → 1  +  Sigmoid',480,184,480,58);
const fusion=box('Condition fusion\n[B, T, 384] → [B, T, 256]',480,281,480,70);
const cat=box('Concatenate  [B, T, 384]',480,389,480,54);
const ln=box('LayerNorm  [B, T, 256]',480,480,480,54);
const gru=box('2-layer BiGRU  (128 / direction)\n[B, T, 256]',480,572,480,70,{size:22});
const mean=box('Frequency mean + Transpose\n[B, T, 96]',480,680,480,68,{size:22});
const cnn=box('3 × ConvFrequencyBlock',480,777,480,54,{size:24});
// Put encoder caption above its group without obscuring the stream.

const mel=box('Log-Mel  [B, T, 64]',480,910,480,56,{size:24});
const wav=box('Waveform y',50,916,165,44,{plain:true,size:26});
const front=box('MelSpectrogram\n+ natural log',230,895,215,80,{size:21});
link(wav,front,'right','left');link(front,mel,'right','left');
[ [mel,cnn],[cnn,mean],[mean,gru],[gru,ln],[ln,cat],[cat,fusion],[fusion,head],[head,out] ].forEach(([a,b])=>link(a,b));
// CNN expansion at the left, read upward.
box('ConvFrequencyBlock',64,407,310,36,{plain:true,size:25});
let c1=box('Conv2D 3 × 3',75,747,285,48);
let c2=box('GroupNorm (8 groups)',75,675,285,48,{size:22});
let c3=box('SiLU',75,603,285,48);
let c4=box('AvgPool2D (1 × 2)',75,531,285,48,{size:22});
let c5=box('Dropout2D (0.2)',75,459,285,48,{size:22});
[[c1,c2],[c2,c3],[c3,c4],[c4,c5]].forEach(([a,b])=>link(a,b));
dotted(365,451,475,777);dotted(365,803,475,831);
box('Channels: 1 → 32 → 64 → 96\nMel bins: 64 → 32 → 16 → 8\nTime resolution T is preserved',58,205,330,132,{plain:true,size:22});
// Category condition stream at the right.
const cond=box('Target-class one-hot\n[B, 527]',1060,749,300,68,{size:23});
const emb=box('Learned embedding\n[527, 128]',1060,633,300,68,{size:23});
const expand=box('Expand across time\n[B, T, 128]',1060,517,300,68,{size:23});
link(cond,emb);link(emb,expand);link(expand,cat,'top','right','elbow');
box('Class conditioning',1040,831,340,34,{plain:true,size:25});
// Fusion expansion at top right.
box('Fusion detail',1045,170,330,32,{plain:true,size:25});
const f1=box('Linear (384 → 256)',1060,370,300,44,{size:22});
const f2=box('SiLU',1060,315,300,38,{size:22});
const f3=box('LayerNorm',1060,260,300,38,{size:22});
const f4=box('Dropout (0.2)',1060,205,300,38,{size:22});
[[f1,f2],[f2,f3],[f3,f4]].forEach(([a,b])=>link(a,b));
dotted(965,281,1050,205);dotted(965,351,1050,414);
box('16 kHz   /   64 Mel bins   /   32 ms window   /   16 ms hop',46,975,1050,24,{plain:true,size:16,align:'left'});
// Remove obsolete bottom caption that would overlap CNN.
// Every module, label, and connector remains native PowerPoint content.
s.speakerNotes.textFrame.setText('Architecture verified against nets/audioset_crnn.py and current run.sh configuration. Front end: datasets/audioset_logmel_dataset.py and README.md. Default channels [32,64,96], 2-layer bidirectional GRU with hidden size 128 per direction, embedding dimension 128, dropout 0.2. CNN tensors [B,1,T,64] -> [B,32,T,32] -> [B,64,T,16] -> [B,96,T,8]. Frequency mean yields [B,96,T], then transpose [B,T,96]. Packed sequences use valid lengths. Frame logits have shape [B,T]; sigmoid yields conditional probabilities. frame_mask excludes batch padding. Training uses real strong annotations and masked balanced BCE, omitted from this forward architecture diagram. Class conditions are single-class one-hot vectors, not multi-hot vectors. B=batch size, T=number of frames. Reference image used for black/white/gray layout and module expansion style only.');
const draft=root+'/.architecture-build/candidate.pptx';await (await PresentationFile.exportPptx(p)).save(draft);
const preview=await p.export({slide:s,format:'png',scale:1.5});await fs.writeFile(root+'/outputs/current_architecture.png',new Uint8Array(await preview.arrayBuffer()));
await finalizePresentation({workspaceDir:root,candidatePath:draft,finalPath:root+'/outputs/current_architecture.pptx',pythonExecutable:'/Users/shenyuanyang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','13716000,9525000'],fontPolicy:{basis:'design',families:[font]},explicitTotalSlideCount:1,verifyArtifactToolImport:true,receiptPath:root+'/.architecture-build/validation.json'});
console.log('Done');
