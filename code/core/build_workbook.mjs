// 对应问题：四问的官方结果表及补充实验。
// 输入：原题模板、results/optimization/workbook_data.json。输出：根目录result_workbook.xlsx。
// 保留原题六张主表的列名；公式汇总、样式和缓存由文档表格库生成。
import fs from 'node:fs/promises';
import {FileBlob,SpreadsheetFile} from '@oai/artifact-tool';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const [payload='results/optimization/workbook_data.json',output='result_workbook.xlsx',audit='results/optimization',mode='write']=process.argv.slice(2);
const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(root+'/data/processed/结果提交模板.xlsx'));
const tabs=JSON.parse(await fs.readFile(root+'/'+payload,'utf8'));
const col=n=>{let s='';while(n){n--;s=String.fromCharCode(65+n%26)+s;n=Math.floor(n/26);}return s};
for(const [name,data] of Object.entries(tabs)){
 const ws=data.header?wb.worksheets.add(name):wb.worksheets.getItem(name);const nc=data.header?.length??data.rows[0].length;const nr=data.rows.length+1;
 if(data.header)ws.getRange(`A1:${col(nc)}1`).values=[data.header];
 if(name==='Q1_单点组批')data.rows=data.rows.map(r=>{const b=r[3].split(';');r[3]=b.map((v,i)=>v+((i+1)%3===0&&i<b.length-1?'\n':'' )).join('; ');return r;});
 ws.getRange(`A2:${col(nc)}${nr}`).values=data.rows;
 const all=ws.getRange(`A1:${col(nc)}${nr}`);all.format.font={name:'Microsoft YaHei',size:11,bold:false};all.format.wrapText=true;all.format.verticalAlignment='center';all.format.horizontalAlignment='center';all.format.rowHeight=32;all.format.columnWidthPx=150;
 ws.getRange(`A1:${col(nc)}1`).format.fill='#DCE6F1';ws.getRange(`A1:${col(nc)}1`).format.font={name:'Microsoft YaHei',size:11,bold:true,color:'#000000'};ws.getRange(`A1:${col(nc)}1`).format.rowHeight=48;
 ws.freezePanes.freezeRows(1);
 if(name.includes('逐箱')){ws.getRange(`A1:A${nr}`).format.columnWidthPx=190;ws.getRange(`D2:D${nr}`).setNumberFormat('0.000');}
 if(name==='Q1_单点组批'){ws.getRange(`D1:D${nr}`).format.columnWidthPx=420;all.format.rowHeight=55;ws.getRange(`E2:I${nr}`).setNumberFormat('0.000');}
 if(name.includes('运输架次')){ws.getRange(`F1:F${nr}`).format.columnWidthPx=220;ws.getRange(`E2:E${nr}`).setNumberFormat('0.000');ws.getRange(`G2:H${nr}`).setNumberFormat('0.000');}
 if(name==='Q3_通信保障'){ws.getRange(`E1:E${nr}`).format.columnWidthPx=210;ws.getRange(`C2:D${nr}`).setNumberFormat('0.000');}
 if(name==='Q3_中继架次'){ws.getRange(`D2:K${nr}`).setNumberFormat('0.000');ws.getRange(`E2:F${nr}`).setNumberFormat('0.00000000');}
 if(name==='Q4_分区配置'){ws.getRange(`C1:C${nr}`).format.columnWidthPx=430;all.format.rowHeight=64;}
 if(name==='Q1_安全载荷')ws.getRange(`B2:D${nr}`).setNumberFormat('0.000');
 if(name==='Q2Q3_资源占用'){ws.getRange(`B1:B${nr}`).format.columnWidthPx=190;ws.getRange(`E2:F${nr}`).setNumberFormat('0.000');ws.getRange(`G2:G${nr}`).setNumberFormat('0%');}
 if(name==='方案比较'){ws.getRange(`A1:A${nr}`).format.columnWidthPx=250;ws.getRange(`G1:G${nr}`).format.columnWidthPx=330;ws.getRange(`B2:D${nr}`).setNumberFormat('0.000');all.format.rowHeight=52;}
 if(name==='Q1_目标优先级'){ws.getRange(`A1:A${nr}`).format.columnWidthPx=390;ws.getRange(`C2:D${nr}`).setNumberFormat('0.000000');all.format.rowHeight=48;}
 if(name==='固定计划扰动边界'){ws.getRange(`A1:A${nr}`).format.columnWidthPx=230;ws.getRange(`B1:F${nr}`).format.columnWidthPx=190;ws.getRange(`B2:F${nr}`).setNumberFormat('0.000000');all.format.rowHeight=48;}
 if(name==='Q4_均衡阈值'){ws.getRange(`A1:A${nr}`).format.columnWidthPx=230;ws.getRange(`E2:F${nr}`).setNumberFormat('0.000000');all.format.rowHeight=42;}
 if(name==='结果与口径'){
  ws.getRange(`A1:A${nr}`).format.columnWidthPx=240;ws.getRange(`B1:B${nr}`).format.columnWidthPx=430;ws.getRange(`C1:C${nr}`).format.columnWidthPx=100;ws.getRange(`D1:D${nr}`).format.columnWidthPx=410;all.format.rowHeight=56;
  const q1=tabs['Q1_单点组批'].rows.length+1,q2=tabs['Q2_运输架次'].rows.length+1,q3=tabs['Q3_运输架次'].rows.length+1,qr=tabs['Q3_中继架次'].rows.length+1;
  const formulas={2:`=COUNTA('Q1_单点组批'!A2:A${q1})`,3:`=SUM('Q1_单点组批'!H2:H${q1})`,4:`=SUM('Q1_单点组批'!G2:G${q1})`,5:`=MAX('Q2_运输架次'!G2:G${q2})`,6:`=SUM('Q2_运输架次'!H2:H${q2})`,8:`=MAX('Q3_运输架次'!G2:G${q3},'Q3_中继架次'!J2:J${qr})`,9:`=SUM('Q3_运输架次'!H2:H${q3},'Q3_中继架次'!K2:K${qr})`};
  for(const [row,f] of Object.entries(formulas))ws.getRange(`B${row}`).formulas=[[f]];
  if(tabs['方案比较']){
   ws.getRange('B22').formulas=[['=1-B21/B8']];
   const qi=data.rows.findIndex(r=>r[0]==='Q2工期界差')+2;
   ws.getRange(`B${qi}`).formulas=[['=1-B21/B5']];ws.getRange(`B${qi}`).setNumberFormat('0.00%');
  }
  for(const row of [3,4,5,6,7,8,9,10,14,15,21])ws.getRange(`B${row}`).setNumberFormat('0.000');for(const row of [11,17,22])ws.getRange(`B${row}`).setNumberFormat('0.00%');
 }
}
wb.recalculate();
if(mode==='check-only'){
 const detail=wb.worksheets.getItem('Q2_运输架次').getRange('H2'),summary=wb.worksheets.getItem('结果与口径').getRange('B6');
 const input=detail.values[0][0],before=summary.values[0][0];detail.values=[[input+1]];wb.recalculate();const changed=summary.values[0][0];
 if(Math.abs(changed-before-1)>1e-8)throw new Error('Summary formula did not respond to the changed sortie energy.');
 detail.values=[[input]];wb.recalculate();const restored=summary.values[0][0];
 if(Math.abs(restored-before)>1e-8)throw new Error('Summary formula did not restore.');
 await fs.writeFile(root+'/'+audit+'/workbook_recalculation_check.json',JSON.stringify({status:'PASS',before,changed,restored,scope:'Disposable in-memory perturbation; exported workbook unchanged.'},null,2));console.log('Recalculation perturbation PASS',before,changed,restored);process.exit(0);
}
await (await SpreadsheetFile.exportXlsx(wb)).save(root+'/'+output);
console.log('Workbook exported with',Object.keys(tabs).length,'sheets');
