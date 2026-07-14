import { chromium } from 'playwright';
import { mkdirSync } from 'fs';

const OUT = 'D:/Barricade/reference/screenshots';
mkdirSync(OUT, { recursive: true });

const browser = await chromium.launch();
const mobile = { width: 375, height: 812 };

// ============ BARRICADE.GG ============
console.log('=== BARRICADE.GG ===');
let ctx = await browser.newContext({ viewport: mobile, colorScheme: 'dark' });
let page = await ctx.newPage();

// Landing - dismiss cookie first
await page.goto('https://barricade.gg', { waitUntil: 'domcontentloaded', timeout: 30000 });
await page.waitForTimeout(3000);
try { await page.locator('text=AGREE').first().click({ timeout: 5000 }); } catch(e) {}
await page.waitForTimeout(1500);
await page.screenshot({ path: `${OUT}/01_barricade_landing.png`, fullPage: true });
console.log('  01_barricade_landing.png');

// Rules page
await page.goto('https://barricade.gg/rules', { waitUntil: 'domcontentloaded', timeout: 30000 });
await page.waitForTimeout(3000);
await page.screenshot({ path: `${OUT}/02_barricade_rules.png`, fullPage: true });
console.log('  02_barricade_rules.png');

// Computer setup modal
await page.goto('https://barricade.gg/computer', { waitUntil: 'domcontentloaded', timeout: 30000 });
await page.waitForTimeout(3000);
await page.screenshot({ path: `${OUT}/03_barricade_computer_setup.png`, fullPage: true });
console.log('  03_barricade_computer_setup.png');

// Local game - board starting position
await page.goto('https://barricade.gg/local', { waitUntil: 'domcontentloaded', timeout: 30000 });
await page.waitForTimeout(5000);
await page.screenshot({ path: `${OUT}/04_barricade_board_start.png`, fullPage: true });
console.log('  04_barricade_board_start.png');

// Play some moves in the local game by clicking cells
// The board uses a grid. Let's find clickable board elements and interact.
// Red pawn starts bottom center - try clicking cells above it to move
try {
  // Find board cells - barricade.gg uses specific classes
  const boardArea = page.locator('[class*="grid"], [class*="board"]').first();
  const box = await boardArea.boundingBox();
  if (box) {
    console.log(`  Board at: ${box.x}, ${box.y}, ${box.width}x${box.height}`);
    const cellW = box.width / 9;
    const cellH = box.height / 9;

    // Red pawn is at row 8 (bottom), col 4 (center). Click row 7 col 4 to move up.
    const clickX = box.x + (4.5 * cellW);
    const clickY = box.y + (7.5 * cellH);
    console.log(`  Clicking cell (7,4) at ${clickX}, ${clickY}`);
    await page.mouse.click(clickX, clickY);
    await page.waitForTimeout(1500);

    // Move blue: click row 1, col 4 to move down
    const clickX2 = box.x + (4.5 * cellW);
    const clickY2 = box.y + (1.5 * cellH);
    await page.mouse.click(clickX2, clickY2);
    await page.waitForTimeout(1500);

    // Move red again
    const clickX3 = box.x + (4.5 * cellW);
    const clickY3 = box.y + (6.5 * cellH);
    await page.mouse.click(clickX3, clickY3);
    await page.waitForTimeout(1500);

    // Move blue again
    const clickX4 = box.x + (4.5 * cellW);
    const clickY4 = box.y + (2.5 * cellH);
    await page.mouse.click(clickX4, clickY4);
    await page.waitForTimeout(1500);

    await page.screenshot({ path: `${OUT}/05_barricade_board_midgame.png`, fullPage: true });
    console.log('  05_barricade_board_midgame.png');

    // Now try placing a wall - click Horizontal button first
    try {
      await page.locator('text=/Horizontal/i').first().click({ timeout: 2000 });
      await page.waitForTimeout(500);
      // Click between rows to place a wall
      const wallX = box.x + (3.5 * cellW);
      const wallY = box.y + (4 * cellH); // between rows 3 and 4
      await page.mouse.click(wallX, wallY);
      await page.waitForTimeout(1500);

      await page.screenshot({ path: `${OUT}/06_barricade_board_with_wall.png`, fullPage: true });
      console.log('  06_barricade_board_with_wall.png');
    } catch(e) { console.log('  wall placement interaction failed:', e.message); }
  }
} catch(e) { console.log('  board interaction failed:', e.message); }

await page.close();
await ctx.close();

// ============ WRONGWAY.APP ============
console.log('\n=== WRONGWAY.APP ===');
ctx = await browser.newContext({ viewport: mobile, colorScheme: 'dark' });
page = await ctx.newPage();

// Set all preferences to skip tutorials/cookies
await page.goto('https://wrongway.app', { waitUntil: 'domcontentloaded', timeout: 30000 });
await page.evaluate(() => {
  localStorage.setItem('ww_dark', '1');
  localStorage.setItem('ww_intro_done', '1');
  localStorage.setItem('ww_tut_done', '1');
  localStorage.setItem('ww_tut_classic', '1');
  localStorage.setItem('ww_tut_duel', '1');
  localStorage.setItem('ww_cookie', '1');
  localStorage.setItem('ww_onboarding', 'done');
});
await page.reload({ waitUntil: 'domcontentloaded', timeout: 30000 });
await page.waitForTimeout(7000);

// Dismiss any remaining overlay
try { await page.locator('text=/Skip/').first().click({ timeout: 2000 }); } catch(e) {}
try { await page.locator('text=/Accept/').first().click({ timeout: 2000 }); } catch(e) {}
await page.waitForTimeout(1000);

// Main menu
await page.screenshot({ path: `${OUT}/07_wrongway_menu.png`, fullPage: true });
console.log('  07_wrongway_menu.png');

// Click into vs Bot -> Easy to start a game
try {
  await page.locator('text=/Bot/i').first().click({ timeout: 3000 });
  await page.waitForTimeout(2000);
  await page.screenshot({ path: `${OUT}/08_wrongway_difficulty.png`, fullPage: true });
  console.log('  08_wrongway_difficulty.png');

  await page.locator('text=/Easy/i').first().click({ timeout: 3000 });
  await page.waitForTimeout(6000); // wait for intro animation

  await page.screenshot({ path: `${OUT}/09_wrongway_board_start.png`, fullPage: true });
  console.log('  09_wrongway_board_start.png');

  // Play some moves - find the board area and click cells
  // In wrongway duel mode: 9x9 board, red starts bottom, blue starts top
  // Red needs to go up, blue goes down
  const boardEl = page.locator('[class*="board"], [class*="grid"]').first();
  const bBox = await boardEl.boundingBox();
  if (bBox) {
    console.log(`  Board at: ${bBox.x}, ${bBox.y}, ${bBox.width}x${bBox.height}`);
    const cW = bBox.width / 9;
    const cH = bBox.height / 9;

    // Red pawn at bottom center (row 8, col 4). Click one cell up.
    await page.mouse.click(bBox.x + 4.5*cW, bBox.y + 7.5*cH);
    await page.waitForTimeout(2500); // wait for bot move

    await page.screenshot({ path: `${OUT}/10_wrongway_board_move1.png`, fullPage: true });
    console.log('  10_wrongway_board_move1.png');

    // Make another move up
    await page.mouse.click(bBox.x + 4.5*cW, bBox.y + 6.5*cH);
    await page.waitForTimeout(2500);

    // Try one more move
    await page.mouse.click(bBox.x + 4.5*cW, bBox.y + 5.5*cH);
    await page.waitForTimeout(2500);

    await page.screenshot({ path: `${OUT}/11_wrongway_board_midgame.png`, fullPage: true });
    console.log('  11_wrongway_board_midgame.png');

    // Try placing a wall - click the horizontal wall tool at bottom
    try {
      const hWallBtn = page.locator('[class*="wtool"]').first();
      if (await hWallBtn.isVisible({ timeout: 2000 })) {
        await hWallBtn.click();
        await page.waitForTimeout(500);
        // Click on the board to place wall
        await page.mouse.click(bBox.x + 3*cW, bBox.y + 4*cH);
        await page.waitForTimeout(2000);
        await page.screenshot({ path: `${OUT}/12_wrongway_board_wall.png`, fullPage: true });
        console.log('  12_wrongway_board_wall.png');
      }
    } catch(e) { console.log('  wall tool failed'); }
  } else {
    console.log('  Could not find board bounding box');
  }
} catch(e) { console.log('  Game start failed:', e.message); }

await page.close();

// --- WRONGWAY light mode menu ---
console.log('\nWrongway light mode...');
ctx = await browser.newContext({ viewport: mobile, colorScheme: 'light' });
page = await ctx.newPage();
await page.goto('https://wrongway.app', { waitUntil: 'domcontentloaded', timeout: 30000 });
await page.evaluate(() => {
  localStorage.setItem('ww_dark', '0');
  localStorage.setItem('ww_intro_done', '1');
  localStorage.setItem('ww_tut_done', '1');
  localStorage.setItem('ww_cookie', '1');
  localStorage.setItem('ww_onboarding', 'done');
});
await page.reload({ waitUntil: 'domcontentloaded', timeout: 30000 });
await page.waitForTimeout(7000);
try { await page.locator('text=/Skip/').first().click({ timeout: 2000 }); } catch(e) {}
await page.waitForTimeout(1000);
await page.screenshot({ path: `${OUT}/13_wrongway_menu_light.png`, fullPage: true });
console.log('  13_wrongway_menu_light.png');

await page.close();
await ctx.close();
await browser.close();
console.log('\n=== ALL DONE ===');
