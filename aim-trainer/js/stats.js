/* Persistence + progress reporting. Everything lives in this browser only. */
var AT = window.AT || {};
AT.store = (function () {
  var SESSIONS = 'aimtrainer.v2.sessions';
  var SETTINGS = 'aimtrainer.v2.settings';
  var MAX = 400;

  var DEFAULTS = {
    sens: 1.0,
    ads: 0.7,
    hand: 'right',
    invert: false,
    tapToFire: true,
    haptics: true,
    size: 1.0
  };

  function read(key, fallback) {
    try {
      var raw = localStorage.getItem(key);
      return raw ? JSON.parse(raw) : fallback;
    } catch (e) {
      return fallback;
    }
  }

  function write(key, value) {
    try {
      localStorage.setItem(key, JSON.stringify(value));
      return true;
    } catch (e) {
      return false;
    }
  }

  var settings = Object.assign({}, DEFAULTS, read(SETTINGS, {}));

  return {
    settings: settings,

    saveSettings: function () {
      write(SETTINGS, settings);
    },

    sessions: function (drillId) {
      var all = read(SESSIONS, []);
      if (!Array.isArray(all)) return [];
      return drillId ? all.filter(function (s) { return s.drill === drillId; }) : all;
    },

    add: function (session) {
      var all = read(SESSIONS, []);
      if (!Array.isArray(all)) all = [];
      all.push(session);
      if (all.length > MAX) all = all.slice(all.length - MAX);
      write(SESSIONS, all);
    },

    clear: function () {
      write(SESSIONS, []);
    },

    /* Best primary result for a drill, honouring lower-is-better metrics. */
    best: function (drillId) {
      var drill = AT.drills.byId(drillId);
      if (!drill) return null;
      var runs = this.sessions(drillId).filter(function (s) { return s.counted; });
      if (!runs.length) return null;
      return runs.reduce(function (a, b) {
        if (drill.primary.lowerIsBetter) return b.primary < a.primary ? b : a;
        return b.primary > a.primary ? b : a;
      });
    },

    /* Mean primary of the runs before the current one, for the verdict line. */
    baseline: function (drillId, excludeTs) {
      var runs = this.sessions(drillId).filter(function (s) {
        return s.counted && s.ts !== excludeTs;
      });
      if (runs.length < 2) return null;
      var recent = runs.slice(-6);
      var sum = recent.reduce(function (acc, s) { return acc + s.primary; }, 0);
      return sum / recent.length;
    }
  };
})();

AT.fmt = {
  primary: function (drill, value) {
    if (value === null || value === undefined || !isFinite(value)) return '—';
    if (drill.primary.decimals === 0) return Math.round(value) + drill.primary.unit;
    return value.toFixed(drill.primary.decimals === undefined ? 1 : drill.primary.decimals) + drill.primary.unit;
  },
  date: function (ts) {
    var d = new Date(ts);
    return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  }
};

/* Small inline sparkline so progress is visible without a chart library. */
AT.sparkline = function (values, lowerIsBetter) {
  var w = 300, h = 44, pad = 4;
  if (!values.length) return '';
  var min = Math.min.apply(null, values);
  var max = Math.max.apply(null, values);
  if (max - min < 1e-9) { max = min + 1; }
  var step = values.length > 1 ? (w - pad * 2) / (values.length - 1) : 0;

  /* On lower-is-better metrics the axis is flipped so that on every chart
     in the app, a line heading upward means you are getting better. */
  var pts = values.map(function (v, i) {
    var x = pad + i * step;
    var norm = (v - min) / (max - min);
    if (lowerIsBetter) norm = 1 - norm;
    var y = h - pad - norm * (h - pad * 2);
    return [x, y];
  });

  var line = pts.map(function (p, i) {
    return (i ? 'L' : 'M') + p[0].toFixed(1) + ' ' + p[1].toFixed(1);
  }).join(' ');

  var area = line + ' L' + pts[pts.length - 1][0].toFixed(1) + ' ' + (h - pad) +
             ' L' + pts[0][0].toFixed(1) + ' ' + (h - pad) + ' Z';

  var first = values[0], last = values[values.length - 1];
  var improving = lowerIsBetter ? last < first : last > first;
  var stroke = improving ? '#3ddc84' : '#ff6b35';
  var lastPt = pts[pts.length - 1];

  return '<svg class="spark" viewBox="0 0 ' + w + ' ' + h + '" preserveAspectRatio="none" aria-hidden="true">' +
    '<path d="' + area + '" fill="' + stroke + '" opacity=".13"/>' +
    '<path d="' + line + '" fill="none" stroke="' + stroke + '" stroke-width="2" ' +
    'stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/>' +
    '<circle cx="' + lastPt[0].toFixed(1) + '" cy="' + lastPt[1].toFixed(1) + '" r="3" fill="' + stroke + '"/>' +
    '</svg>';
};

window.AT = AT;
