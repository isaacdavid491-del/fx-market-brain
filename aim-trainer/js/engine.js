/* Render loop, touch input and the shared game context handed to each drill. */
var AT = window.AT || {};

AT.Engine = function (opts) {
  var canvas = opts.canvas;
  var stage = opts.stage;
  var fireBtn = opts.fireBtn;
  var countdownEl = opts.countdown;
  var ctx = canvas.getContext('2d', { alpha: false });

  var drill = null;
  var settings = null;
  var raf = 0;
  var lastFrame = 0;
  var phase = 'idle';        /* idle | countdown | running | done */
  var countdownLeft = 0;
  var fx = [];
  var missFlash = 0;
  var shotFlash = 0;
  var pointers = {};
  var lookId = null;

  var g = {
    w: 0, h: 0, cx: 0, cy: 0,
    view: { x: 0, y: 0 },
    t: 0,
    elapsed: 0,
    firing: false,
    viewDelta: 0,
    targetScale: 1,
    hudSub: '',
    state: null
  };

  /* Debug handle: lets tests and the console inspect live drill state. */
  AT.__g = g;

  /* ---------- sizing ---------- */

  function resize() {
    var dpr = Math.min(window.devicePixelRatio || 1, 3);
    var rect = stage.getBoundingClientRect();
    g.w = Math.max(1, rect.width);
    g.h = Math.max(1, rect.height);
    g.cx = g.w / 2;
    g.cy = g.h / 2;
    canvas.width = Math.round(g.w * dpr);
    canvas.height = Math.round(g.h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  /* ---------- context API for drills ---------- */

  g.hit = function (screenPos) {
    burst(screenPos.x, screenPos.y, '#3ddc84');
    haptic(12);
  };

  g.missFx = function () {
    missFlash = 1;
    haptic(35);
  };

  g.shotFx = function () {
    shotFlash = 1;
  };

  g.finishNow = function () {
    if (phase === 'running') finish();
  };

  function haptic(ms) {
    if (!settings || !settings.haptics) return;
    if (navigator.vibrate) {
      try { navigator.vibrate(ms); } catch (e) { /* ignore */ }
    }
  }

  function burst(x, y, color) {
    for (var i = 0; i < 9; i++) {
      var a = Math.random() * Math.PI * 2;
      var sp = 60 + Math.random() * 170;
      fx.push({
        x: x, y: y,
        vx: Math.cos(a) * sp, vy: Math.sin(a) * sp,
        life: 0.42, max: 0.42, color: color
      });
    }
  }

  /* ---------- input ---------- */

  function localPoint(ev) {
    var r = stage.getBoundingClientRect();
    return { x: ev.clientX - r.left, y: ev.clientY - r.top };
  }

  function onPointerDown(ev) {
    if (phase !== 'running') return;
    ev.preventDefault();
    var p = localPoint(ev);

    if (drill.mode === 'tap') {
      if (drill.onTap) drill.onTap(g, p.x, p.y);
      return;
    }

    pointers[ev.pointerId] = { x: p.x, y: p.y, sx: p.x, sy: p.y, t: performance.now(), moved: 0 };

    if (lookId === null) {
      lookId = ev.pointerId;
    } else if (drill.onFire) {
      /* A second finger anywhere is an alternate trigger. */
      drill.onFire(g);
    }
    try { stage.setPointerCapture(ev.pointerId); } catch (e) { /* ignore */ }
  }

  function onPointerMove(ev) {
    if (phase !== 'running' || drill.mode === 'tap') return;
    var rec = pointers[ev.pointerId];
    if (!rec) return;
    ev.preventDefault();

    var p = localPoint(ev);
    var dx = p.x - rec.x;
    var dy = p.y - rec.y;
    rec.x = p.x; rec.y = p.y;
    rec.moved += Math.abs(dx) + Math.abs(dy);

    if (ev.pointerId !== lookId) return;

    var sens = settings.sens * (drill.scoped ? settings.ads : 1);
    g.view.x += dx * sens;
    g.view.y += (settings.invert ? -dy : dy) * sens;
    g.viewDelta += Math.hypot(dx, dy) * sens;
  }

  function onPointerUp(ev) {
    var rec = pointers[ev.pointerId];
    delete pointers[ev.pointerId];
    if (ev.pointerId === lookId) lookId = null;
    if (phase !== 'running' || !rec || drill.mode === 'tap') return;

    /* A quick stationary touch in the look area counts as a shot. */
    var quick = performance.now() - rec.t < 260;
    if (settings.tapToFire && quick && rec.moved < 10 && drill.onFire && drill.id !== 'recoil') {
      drill.onFire(g);
    }
  }

  function onFireDown(ev) {
    ev.preventDefault();
    if (phase !== 'running') return;
    fireBtn.classList.add('down');
    g.firing = true;
    if (drill.onFire) drill.onFire(g);
  }

  function onFireUp(ev) {
    if (ev) ev.preventDefault();
    fireBtn.classList.remove('down');
    g.firing = false;
  }

  stage.addEventListener('pointerdown', onPointerDown, { passive: false });
  stage.addEventListener('pointermove', onPointerMove, { passive: false });
  stage.addEventListener('pointerup', onPointerUp, { passive: false });
  stage.addEventListener('pointercancel', onPointerUp, { passive: false });
  fireBtn.addEventListener('pointerdown', onFireDown, { passive: false });
  fireBtn.addEventListener('pointerup', onFireUp, { passive: false });
  fireBtn.addEventListener('pointercancel', onFireUp, { passive: false });
  stage.addEventListener('contextmenu', function (e) { e.preventDefault(); });
  window.addEventListener('resize', function () { if (phase !== 'idle') resize(); });
  window.addEventListener('orientationchange', function () { setTimeout(resize, 250); });

  /* ---------- drawing ---------- */

  function drawCrosshair() {
    var x = g.cx, y = g.cy;
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,.88)';
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.moveTo(x - 14, y); ctx.lineTo(x - 5, y);
    ctx.moveTo(x + 5, y); ctx.lineTo(x + 14, y);
    ctx.moveTo(x, y - 14); ctx.lineTo(x, y - 5);
    ctx.moveTo(x, y + 5); ctx.lineTo(x, y + 14);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(x, y, 1.6, 0, Math.PI * 2);
    ctx.fillStyle = '#fff';
    ctx.fill();
    ctx.restore();
  }

  function drawFx(dt) {
    for (var i = fx.length - 1; i >= 0; i--) {
      var p = fx[i];
      p.life -= dt;
      if (p.life <= 0) { fx.splice(i, 1); continue; }
      p.x += p.vx * dt;
      p.y += p.vy * dt;
      p.vy += 240 * dt;
      ctx.save();
      ctx.globalAlpha = Math.max(0, p.life / p.max);
      ctx.fillStyle = p.color;
      ctx.fillRect(p.x - 2, p.y - 2, 4, 4);
      ctx.restore();
    }

    if (missFlash > 0) {
      missFlash = Math.max(0, missFlash - dt * 4);
      var grd = ctx.createRadialGradient(g.cx, g.cy, Math.min(g.w, g.h) * 0.25, g.cx, g.cy, Math.max(g.w, g.h) * 0.7);
      grd.addColorStop(0, 'rgba(255,77,94,0)');
      grd.addColorStop(1, 'rgba(255,77,94,' + (0.45 * missFlash).toFixed(3) + ')');
      ctx.fillStyle = grd;
      ctx.fillRect(0, 0, g.w, g.h);
    }

    if (shotFlash > 0) {
      shotFlash = Math.max(0, shotFlash - dt * 12);
      ctx.save();
      ctx.globalAlpha = shotFlash * 0.25;
      ctx.fillStyle = '#ffb03a';
      ctx.beginPath();
      ctx.arc(g.cx, g.cy, 22, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
  }

  /* ---------- loop ---------- */

  function frame(now) {
    raf = requestAnimationFrame(frame);
    var dt = Math.min(0.05, (now - lastFrame) / 1000);
    if (!(dt > 0)) dt = 0;
    lastFrame = now;

    ctx.fillStyle = '#070a0e';
    ctx.fillRect(0, 0, g.w, g.h);

    if (phase === 'countdown') {
      countdownLeft -= dt;
      var n = Math.ceil(countdownLeft);
      countdownEl.textContent = n > 0 ? String(n) : 'GO';
      if (countdownLeft <= -0.35) {
        countdownEl.classList.remove('show');
        phase = 'running';
        g.t = 0;
        g.elapsed = 0;
        drill.init(g);
      }
      if (drill.mode === 'aim') drawCrosshair();
      return;
    }

    if (phase !== 'running') return;

    g.t += dt;
    g.elapsed = g.t;

    /* Drills read viewDelta (thumb travel since the last frame), so it is
       cleared only after update has had a chance to consume it. */
    drill.update(g, dt);
    g.viewDelta = 0;
    drill.draw(g, ctx);
    if (drill.mode === 'aim') drawCrosshair();
    drawFx(dt);

    if (!drill.untimed) {
      var left = Math.max(0, drill.duration - g.t);
      opts.onTick(left, g.hudSub);
      if (left <= 0) finish();
    } else {
      opts.onTick(g.t, g.hudSub);
    }
  }

  function finish() {
    phase = 'done';
    cancelAnimationFrame(raf);
    onFireUp();
    pointers = {};
    lookId = null;
    var result = drill.finish(g);
    result.drill = drill.id;
    result.ts = Date.now();
    result.sens = settings.sens;
    result.duration = Math.round(g.elapsed);
    opts.onFinish(drill, result);
  }

  return {
    start: function (d, s) {
      drill = d;
      settings = s;
      resize();
      g.view.x = 0; g.view.y = 0;
      g.t = 0; g.elapsed = 0;
      g.viewDelta = 0;
      g.firing = false;
      g.targetScale = s.size;
      g.hudSub = '';
      fx = [];
      missFlash = 0;
      shotFlash = 0;
      pointers = {};
      lookId = null;

      fireBtn.classList.toggle('left', s.hand === 'left');
      fireBtn.classList.toggle('hidden', d.mode === 'tap');

      countdownEl.classList.add('show');
      countdownEl.textContent = '3';
      countdownLeft = 3;
      phase = 'countdown';
      lastFrame = performance.now();
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(frame);
    },

    stop: function () {
      phase = 'idle';
      cancelAnimationFrame(raf);
      onFireUp();
      countdownEl.classList.remove('show');
      pointers = {};
      lookId = null;
    },

    isRunning: function () { return phase === 'running' || phase === 'countdown'; }
  };
};

window.AT = AT;
