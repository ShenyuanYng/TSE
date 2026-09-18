import fs from 'node:fs/promises';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const p=await PresentationFile.importPptx(await FileBlob.load('/Users/shenyuanyang/Desktop/new_example_code/outputs/current_architecture.pptx'));
const b=await p.export({slide:p.slides.items[0],format:'png',scale:1.5});
await fs.writeFile('/Users/shenyuanyang/Desktop/new_example_code/outputs/current_architecture.png',new Uint8Array(await b.arrayBuffer()));
