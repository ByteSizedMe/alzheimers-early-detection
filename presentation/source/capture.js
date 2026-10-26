const { chromium } = require('playwright');
const OUT = '/tmp/claude-0/deck/shots/';
(async () => {
  const b = await chromium.launch({args:['--autoplay-policy=no-user-gesture-required']});
  const ctx = await b.newContext({viewport:{width:1440,height:900}, deviceScaleFactor:2});
  const p = await ctx.newPage();
  const errs=[]; p.on('pageerror',e=>errs.push(String(e)));
  const settle = async (ms=2600)=>p.waitForTimeout(ms);
  const shot = async (sel, name) => { const el = p.locator(sel).first(); await el.scrollIntoViewIfNeeded(); await el.screenshot({path: OUT+name+'.png'}); };
  await p.goto('http://localhost:8000'); await settle(2500);
  await p.screenshot({path: OUT+'home.png'});                                  // upload screen with samples
  // ---- MRI: AD sample
  await p.locator('.sample[data-name="AD_alzheimers_01.png"]').click();
  await p.waitForSelector('#results:not([hidden])',{timeout:60000}); await settle(3000);
  await p.evaluate(()=>window.scrollTo(0,0)); await p.waitForTimeout(400);
  await p.screenshot({path: OUT+'mri_ad_full.png'});
  await shot('#verdict','mri_ad_verdict');
  await shot('#stepper','mri_stepper');
  await shot('#results .grid-2 .panel:nth-child(1)','mri_attribution');
  await shot('#results .grid-2 .panel:nth-child(2)','mri_probs');
  await shot('#audit-panel','mri_audit');
  // views
  await p.locator('.seg-btn[data-view="overlay"]').click(); await p.waitForTimeout(700); await shot('#results .grid-2 .panel:nth-child(1)','mri_overlay');
  await p.locator('.seg-btn[data-view="heatmap"]').click(); await p.waitForTimeout(700); await shot('#results .grid-2 .panel:nth-child(1)','mri_heatmap');
  await p.locator('.seg-btn[data-view="compare"]').click(); await p.waitForTimeout(500);
  // ---- known misread
  await p.locator('#reset-btn').click(); await p.waitForTimeout(800);
  await p.locator('.sample[data-name="MISREAD_true-EMCI_predicted-NC.png"]').click();
  await p.waitForSelector('#results:not([hidden])',{timeout:60000}); await settle(3000);
  await p.evaluate(()=>window.scrollTo(0,0)); await p.waitForTimeout(300);
  await shot('#verdict','mri_misread_verdict'); await shot('#results .grid-2 .panel:nth-child(2)','mri_misread_probs'); await shot('#audit-panel','mri_misread_audit');
  await p.screenshot({path: OUT+'mri_misread_full.png'});
  // ---- counterfactual: normal scan, ask "what would make this AD"
  await p.locator('#reset-btn').click(); await p.waitForTimeout(800);
  await p.locator('.sample[data-name="NC_normal-cognition_01.png"]').click();
  await p.waitForSelector('#results:not([hidden])',{timeout:60000}); await settle(3000);
  await shot('#verdict','mri_nc_verdict'); await shot('#results .grid-2 .panel:nth-child(1)','mri_nc_attribution');
  await p.selectOption('#target-select','AD'); await settle(5000);
  await shot('#results .grid-2 .panel:nth-child(1)','mri_counterfactual');
  // ---- handwriting
  await p.locator('.tab[data-view=hw]').click(); await settle(900);
  await p.screenshot({path: OUT+'hw_empty.png'});
  const box = await p.locator('#hw-canvas').boundingBox();
  const cx = box.x+box.width/2, cy = box.y+box.height/2, R = box.height*0.38;
  await p.mouse.move(cx, cy); await p.mouse.down();
  const N = 420;
  for (let i=1;i<=N;i++){ const th=i/N*3*2*Math.PI, r=R*i/N+(Math.random()-0.5)*2;
    await p.mouse.move(cx+r*Math.cos(th)+(Math.random()-0.5)*1.5, cy+r*Math.sin(th)+(Math.random()-0.5)*1.5); await p.waitForTimeout(12+Math.random()*8); }
  await p.mouse.up(); await p.waitForTimeout(400);
  await shot('#hw-stage','hw_live_canvas');
  await p.locator('#hw-analyse').click(); await p.waitForSelector('#hw-results:not([hidden])',{timeout:30000}); await settle(3000);
  await shot('#hw-verdict','hw_live_verdict');
  await p.locator('#hw-reset').click(); await p.waitForTimeout(700);
  for (const prof of ['typical','elevated']) {
    await p.locator(`.hw-demo[data-profile=${prof}]`).click();
    await p.waitForSelector('#hw-results:not([hidden])',{timeout:90000}); await settle(3200);
    await shot('#hw-stage','hw_'+prof+'_canvas'); await shot('#hw-verdict','hw_'+prof+'_verdict');
    await shot('#hw-results .grid-2 .panel:nth-child(1)','hw_'+prof+'_trace'); await shot('#hw-results .grid-2 .panel:nth-child(2)','hw_'+prof+'_features');
    if (prof==='elevated') await shot('#hw-fusion','hw_fusion');
    await p.locator('#hw-reset').click(); await p.waitForTimeout(700);
  }
  await b.close(); console.log('errors', errs);
})();
