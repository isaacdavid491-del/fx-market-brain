/* Screen routing, settings UI, results and progress reporting. */
(function () {
  var AT = window.AT;
  var S = AT.store.settings;
  var $ = function (id) { return document.getElementById(id); };

  var screens = {
    menu: $('screen-menu'),
    brief: $('screen-brief'),
    game: $('screen-game'),
    result: $('screen-result')
  };

  var currentDrill = null;
  var lastResult = null;
  var wakeLock = null;

  function show(name) {
    Object.keys(screens).forEach(function (k) {
      screens[k].classList.toggle('active', k === name);
    });
    if (name !== 'game') releaseWake();
    window.scrollTo(0, 0);
  }

  function requestWake() {
    if (!navigator.wakeLock) return;
    navigator.wakeLock.request('screen').then(function (l) { wakeLock = l; })
      .catch(function () { /* not critical */ });
  }

  function releaseWake() {
    if (wakeLock) {
      try { wakeLock.release(); } catch (e) { /* ignore */ }
      wakeLock = null;
    }
  }

  /* ================= engine ================= */

  var engine = AT.Engine({
    canvas: $('canvas'),
    stage: $('stage'),
    fireBtn: $('fire'),
    countdown: $('countdown'),
    onTick: function (seconds, sub) {
      $('hud-timer').textContent = seconds.toFixed(1);
      $('hud-sub').textContent = sub || '';
    },
    onFinish: function (drill, result) {
      lastResult = result;
      AT.store.add({
        drill: result.drill,
        ts: result.ts,
        primary: result.primary,
        counted: !!result.counted,
        sens: result.sens,
        duration: result.duration
      });
      renderResult(drill, result);
      show('result');
    }
  });

  /* ================= drill list ================= */

  function renderDrillList() {
    var html = AT.drills.all.map(function (d) {
      var best = AT.store.best(d.id);
      var bestTxt = best ? AT.fmt.primary(d, best.primary) : '—';
      return '<button class="drill" data-drill="' + d.id + '">' +
        '<span class="glyph"><svg viewBox="0 0 24 24">' + AT.drills.icon(d.icon) + '</svg></span>' +
        '<span class="meta"><b>' + d.name + '</b><small>' + d.blurb + '</small></span>' +
        '<span class="best">best<strong>' + bestTxt + '</strong></span>' +
        '</button>';
    }).join('');
    $('drill-list').innerHTML = html;
  }

  $('drill-list').addEventListener('click', function (e) {
    var btn = e.target.closest('.drill');
    if (!btn) return;
    openBrief(AT.drills.byId(btn.dataset.drill));
  });

  /* ================= brief ================= */

  function openBrief(drill) {
    currentDrill = drill;
    $('brief-name').textContent = drill.name;
    $('brief-desc').textContent = drill.blurb;
    $('brief-trains').textContent = drill.trains;
    $('brief-how').innerHTML = drill.how.map(function (h) { return '<li>' + h + '</li>'; }).join('');
    $('brief-len').textContent = drill.untimed ? '6 rounds' : drill.duration + 's';
    $('brief-sens').textContent = drill.mode === 'tap' ? 'n/a' : S.sens.toFixed(2);
    var best = AT.store.best(drill.id);
    $('brief-best').textContent = best ? AT.fmt.primary(drill, best.primary) : '—';
    show('brief');
  }

  $('brief-back').addEventListener('click', function () { show('menu'); });
  $('brief-start').addEventListener('click', startDrill);

  function startDrill() {
    show('game');
    requestWake();
    /* Let the layout settle before the engine measures the stage. */
    requestAnimationFrame(function () {
      requestAnimationFrame(function () { engine.start(currentDrill, S); });
    });
  }

  $('hud-quit').addEventListener('click', function () {
    engine.stop();
    show('menu');
    renderDrillList();
  });

  /* ================= results ================= */

  function renderResult(drill, result) {
    $('res-title').textContent = drill.name + ' complete';
    $('res-primary').textContent = AT.fmt.primary(drill, result.primary).replace(drill.primary.unit, '');
    $('res-primary-label').textContent = drill.primary.label +
      (drill.primary.unit ? ' (' + drill.primary.unit + ')' : '');

    var verdict = $('res-verdict');
    if (!result.counted) {
      verdict.className = 'verdict flat';
      verdict.textContent = 'Not enough data to count this run';
    } else {
      var base = AT.store.baseline(drill.id, result.ts);
      var best = AT.store.best(drill.id);
      if (best && best.ts === result.ts && AT.store.sessions(drill.id).filter(function (s) { return s.counted; }).length > 1) {
        verdict.className = 'verdict up';
        verdict.textContent = 'New personal best';
      } else if (base) {
        var better = drill.primary.lowerIsBetter ? result.primary < base : result.primary > base;
        var delta = Math.abs((result.primary - base) / base * 100);
        verdict.className = 'verdict ' + (better ? 'up' : 'flat');
        verdict.textContent = (better ? 'Up ' : 'Down ') + delta.toFixed(0) + '% on your recent average';
      } else {
        verdict.className = 'verdict flat';
        verdict.textContent = 'Baseline recorded — run it twice more to see a trend';
      }
    }

    $('res-metrics').innerHTML = result.metrics.map(function (m) {
      return '<div class="metric"><span class="k">' + m.k + '</span><span class="v">' + m.v + '</span></div>';
    }).join('');

    $('res-coach').innerHTML = '<h3>What to fix</h3><p>' + drill.coach(result) + '</p>';
  }

  $('res-again').addEventListener('click', startDrill);
  $('res-menu').addEventListener('click', function () {
    show('menu');
    renderDrillList();
    renderStats();
  });

  /* ================= settings ================= */

  function bindRange(inputId, outId, key, decimals) {
    var input = $(inputId), out = $(outId);
    input.value = S[key];
    out.textContent = Number(S[key]).toFixed(decimals);
    input.addEventListener('input', function () {
      S[key] = parseFloat(input.value);
      out.textContent = S[key].toFixed(decimals);
      AT.store.saveSettings();
      syncSensChips();
    });
  }

  function bindToggle(inputId, key) {
    var input = $(inputId);
    input.checked = !!S[key];
    input.addEventListener('change', function () {
      S[key] = input.checked;
      AT.store.saveSettings();
    });
  }

  function bindSelect(inputId, key, numeric) {
    var input = $(inputId);
    input.value = S[key];
    input.addEventListener('change', function () {
      S[key] = numeric ? parseFloat(input.value) : input.value;
      AT.store.saveSettings();
    });
  }

  function syncSensChips() {
    var chips = $('sens-presets').querySelectorAll('.chip');
    for (var i = 0; i < chips.length; i++) {
      chips[i].classList.toggle('on', Math.abs(parseFloat(chips[i].dataset.sens) - S.sens) < 0.001);
    }
  }

  bindRange('set-sens', 'out-sens', 'sens', 2);
  bindRange('set-ads', 'out-ads', 'ads', 2);
  bindToggle('set-invert', 'invert');
  bindToggle('set-taptofire', 'tapToFire');
  bindToggle('set-haptics', 'haptics');
  bindSelect('set-hand', 'hand', false);
  bindSelect('set-size', 'size', true);
  syncSensChips();

  $('sens-presets').addEventListener('click', function (e) {
    var chip = e.target.closest('.chip');
    if (!chip) return;
    S.sens = parseFloat(chip.dataset.sens);
    $('set-sens').value = S.sens;
    $('out-sens').textContent = S.sens.toFixed(2);
    AT.store.saveSettings();
    syncSensChips();
  });

  /* ================= tabs ================= */

  document.querySelector('.tabs').addEventListener('click', function (e) {
    var tab = e.target.closest('.tab');
    if (!tab) return;
    var name = tab.dataset.tab;
    document.querySelectorAll('.tab').forEach(function (t) {
      t.classList.toggle('active', t === tab);
    });
    document.querySelectorAll('.tabpanel').forEach(function (p) {
      p.classList.toggle('active', p.dataset.panel === name);
    });
    if (name === 'stats') renderStats();
  });

  /* ================= progress ================= */

  function sensReport(drill, runs) {
    var buckets = {};
    runs.forEach(function (r) {
      var key = (Math.round(r.sens * 10) / 10).toFixed(1);
      (buckets[key] = buckets[key] || []).push(r.primary);
    });
    var keys = Object.keys(buckets).filter(function (k) { return buckets[k].length >= 2; });
    if (keys.length < 2) return '';

    var rows = keys.map(function (k) {
      var vals = buckets[k];
      var mean = vals.reduce(function (a, b) { return a + b; }, 0) / vals.length;
      return { sens: k, n: vals.length, mean: mean };
    });

    var best = rows.reduce(function (a, b) {
      if (drill.primary.lowerIsBetter) return b.mean < a.mean ? b : a;
      return b.mean > a.mean ? b : a;
    });

    rows.sort(function (a, b) { return parseFloat(a.sens) - parseFloat(b.sens); });

    return '<table class="sens"><thead><tr><th>Sens</th><th>Runs</th><th>Average</th></tr></thead><tbody>' +
      rows.map(function (r) {
        return '<tr class="' + (r === best ? 'best' : '') + '"><td>' + r.sens + '</td><td>' + r.n +
          '</td><td>' + AT.fmt.primary(drill, r.mean) + '</td></tr>';
      }).join('') +
      '</tbody></table>' +
      '<p class="hint" style="margin-top:8px">Your best average on this drill came at sensitivity ' +
      best.sens + '.</p>';
  }

  function renderStats() {
    var body = $('stats-body');
    var any = false;
    var html = '';

    AT.drills.all.forEach(function (d) {
      var runs = AT.store.sessions(d.id).filter(function (s) { return s.counted; });
      if (!runs.length) return;
      any = true;
      var values = runs.slice(-20).map(function (s) { return s.primary; });
      var best = runs.reduce(function (a, b) {
        if (d.primary.lowerIsBetter) return b.primary < a.primary ? b : a;
        return b.primary > a.primary ? b : a;
      });
      var recent = values.slice(-5);
      var avg = recent.reduce(function (a, b) { return a + b; }, 0) / recent.length;

      html += '<div class="stat-block">' +
        '<div class="stat-head"><b>' + d.name + '</b><span>' + runs.length + ' runs</span></div>' +
        AT.sparkline(values, d.primary.lowerIsBetter) +
        '<div class="stat-foot">' +
        '<span>best <b>' + AT.fmt.primary(d, best.primary) + '</b></span>' +
        '<span>last 5 <b>' + AT.fmt.primary(d, avg) + '</b></span>' +
        '<span>since <b>' + AT.fmt.date(runs[0].ts) + '</b></span>' +
        '</div>';

      if (d.mode === 'aim' && runs.length >= 4) html += sensReport(d, runs);
      html += '</div>';
    });

    if (!any) {
      body.innerHTML = '<div class="empty">No runs yet.<br>Finish a drill and your scores, trend line and best sensitivity show up here.</div>';
      return;
    }

    html += '<button class="danger" id="wipe">Erase all saved runs</button>';
    body.innerHTML = html;

    $('wipe').addEventListener('click', function () {
      if (confirm('Erase every saved run on this device?')) {
        AT.store.clear();
        renderStats();
        renderDrillList();
      }
    });
  }

  /* ================= boot ================= */

  renderDrillList();
  renderStats();

  document.addEventListener('visibilitychange', function () {
    if (document.hidden && engine.isRunning()) {
      engine.stop();
      show('menu');
    }
  });

  if ('serviceWorker' in navigator && location.protocol.indexOf('http') === 0) {
    navigator.serviceWorker.register('sw.js').catch(function () { /* offline mode optional */ });
  }
})();
