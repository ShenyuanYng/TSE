import fs from 'node:fs/promises';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
import {resolvePresentationFont,finalizePresentation} from '/Users/shenyuanyang/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.61513/skills/presentations/container_tools/artifact_tool_utils.mjs';
const root='/Users/shenyuanyang/Desktop/new_example_code';
const skill='/Users/shenyuanyang/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.61513/skills/presentations';
const font=resolvePresentationFont({fontFamily:'Arial'});
const p=Presentation.create({slideSize:{width:1600,height:1050}});const s=p.slides.add();s.background.fill='#FFFFFF';
function box(text,x,y,w,h,opts={}){let a=s.shapes.add({geometry:opts.plain?'textbox':'roundRect',position:{left:x,top:y,width:w,height:h},fill:opts.plain?'none':(opts.fill||'#FFFFFF'),line:{fill:opts.plain?'none':'#111111',width:opts.plain?0:2.5},borderRadius:opts.plain?0:10});a.text=text;a.text.style={typeface:font,fontSize:opts.size||23,color:'#111111',alignment:opts.align||'center',verticalAlignment:'middle',autoFit:'none',insets:{left:8,right:8,top:3,bottom:3},bold:opts.bold||false};return a;}
function bg(x,y,w,h,color){return s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill:color,line:{fill:'none',width:0}});}
function link(a,b,from='top',to='bottom',kind='straight',dash=false){let c=s.shapes.connect(a,b,{kind,fromSide:from,toSide:to,line:{fill:'#111111',width:dash?1.6:2,style:dash?'dashed':'solid'},tail:{type:dash?'none':'triangle',width:'med',length:'med'}});c.bringToFront();return c;}
function dotted(x1,y1,x2,y2){const dx=x2-x1,dy=y2-y1,len=Math.hypot(dx,dy);s.shapes.add({geometry:'line',position:{left:(x1+x2-len)/2,top:(y1+y2)/2,width:len,height:0,rotation:Math.atan2(dy,dx)*180/Math.PI},fill:'none',line:{fill:'#111111',width:1.5,style:'dashed'}});}
box('Teacher–student framework with loss masking',45,26,1500,60,{plain:true,size:36,align:'left'});
bg(40,180,510,785,'#F2F2F2');bg(1050,180,510,785,'#F2F2F2');bg(590,335,420,530,'#D9D9D9');
box('1. Teacher & label screening',60,196,470,45,{plain:true,size:28,align:'left'});
box('2. Student\ntraining',1070,194,220,66,{plain:true,size:25,align:'left'});
const flag=box('Flag top-scored implicit negatives',75,340,440,65,{size:24});
const rank=box('Rank implicit negatives per class',75,452,440,65,{size:24});
const score=box('Predict training-set scores',75,564,440,65,{size:25});
const teacher=box('Train ResNet-50 teacher\nOriginal labels + BCE',75,676,440,76,{size:25});
const feature=box('Log-Mel patches',75,799,440,57,{size:25});
const data=box('Training audio + original labels y',75,907,440,57,{size:24});
[[data,feature],[feature,teacher],[teacher,score],[score,rank],[rank,flag]].forEach(([a,b])=>link(a,b));
const enhanced=box('Enhanced labels\npositive / negative / ignored',615,340,370,86,{size:23});
const mask=box('Targets y + binary mask M\n0: flagged    1: otherwise',615,250,370,66,{size:23});
link(flag,enhanced,'right','left');link(enhanced,mask);
const loss=box('Masked BCE\n−Σc [yc log(pc)\n+ Mc(1 − yc) log(1 − pc)]',1080,272,450,132,{size:25});
link(mask,loss,'right','left','elbow');
const output=box('Trained student',1115,103,380,56,{size:27});link(loss,output);
const probs=box('Student class probabilities pc',1080,470,450,66,{size:25});
const student=box('Student CNN\nResNet-50 or MobileNetV1',1080,611,450,84,{size:25});
const sf=box('Log-Mel patches',1080,790,450,66,{size:25});
const sd=box('Training audio',1080,907,450,57,{size:25});
[[sd,sf],[sf,student],[student,probs],[probs,loss]].forEach(([a,b])=>link(a,b));
box('Backpropagation',1320,163,225,32,{plain:true,size:18});
box('Per-class threshold tc',613,520,370,40,{plain:true,size:25});
box('Select a percentile of teacher\nscores on implicit negatives',610,570,380,67,{plain:true,size:22});
box('Flag when score > tc',615,672,370,56,{size:24});
box('Explicit labels are retained',610,761,380,52,{plain:true,size:22});
dotted(520,462,585,518);dotted(520,507,585,820);
box('Front end: 64 Mel bands, 25 ms Hann window, 10 ms hop; 96-frame patches, 50% overlap',50,980,1490,30,{plain:true,size:19,align:'left'});
box('Fonseca et al., IEEE SPL 27 (2020) 1235–1239.  DOI: 10.1109/LSP.2020.3006378',50,1014,1490,26,{plain:true,size:17,align:'left'});
s.speakerNotes.textFrame.setText('Source: https://arxiv.org/pdf/2005.00878 (v2), Fig. 1, Section III and Sections IV–V. Redrawn method diagram. Mask applies to the negative BCE term. Internal AudioSet includes explicit/implicit label metadata.');
const draft=root+'/.architecture-build/paper-candidate.pptx';await (await PresentationFile.exportPptx(p)).save(draft);
await finalizePresentation({workspaceDir:root,candidatePath:draft,finalPath:root+'/outputs/teacher_student_loss_masking_final.pptx',pythonExecutable:'/Users/shenyuanyang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','15240000,10001250'],fontPolicy:{basis:'design',families:[font]},explicitTotalSlideCount:1,verifyArtifactToolImport:true,receiptPath:root+'/.architecture-build/paper-validation-final.json'});
const preview=await p.export({slide:s,format:'png',scale:1.3});await fs.writeFile(root+'/outputs/teacher_student_loss_masking.png',new Uint8Array(await preview.arrayBuffer()));
console.log('Done');
