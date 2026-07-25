// figtree-news app.js — extracted from base.html for better separation

function doHeaderSearch() {
  const q = document.getElementById('search-input').value.trim();
  const range = document.getElementById('range-select').value;
  if (q || range !== 'all') {
    window.location.href = `/search?q=${encodeURIComponent(q)}&range=${range}`;
  }
}

function toggleSettings() {
  const panel = document.getElementById('settings-panel');
  const backdrop = document.getElementById('settings-backdrop');
  const open = panel.classList.toggle('open');
  backdrop.style.display = open ? 'block' : 'none';
  document.body.style.overflow = open ? 'hidden' : '';
}

document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape' && document.getElementById('settings-panel').classList.contains('open')) {
    toggleSettings();
  }
});

// Live clock
function updateClock() {
  document.getElementById('live-clock').textContent = new Date().toISOString().slice(11, 19) + 'Z';
}
setInterval(updateClock, 1000);
updateClock();

// Stats polling
let _autoRefreshEnabled = true;
function toggleAutoRefresh() {
  _autoRefreshEnabled = document.getElementById('auto-refresh').checked;
}

async function updateStats() {
  try {
    const resp = await fetch('/api/stats');
    const s = await resp.json();
    document.getElementById('stat-articles').textContent = s.articles;
    document.getElementById('stat-narratives').textContent = s.narratives;
    document.getElementById('stat-sources').textContent = s.sources;
    document.getElementById('stat-brief').textContent = s.has_brief ? 'Yes' : 'No';
  } catch (e) {}
}
updateStats();
setInterval(updateStats, 30000);

// WebSocket + crawl control
var WS_URL = 'ws://' + location.host + '/ws';
var ws = null;
var crawlRunning = false;
var _lastArticleCount = 0;
var _contentRefreshTimer = null;

function connectWS() {
  ws = new WebSocket(WS_URL);
  ws.onmessage = function(e) {
    var msg = JSON.parse(e.data);
    if (msg.type === 'crawl_status') updateCrawlUI(msg.data);
    if (msg.type === 'content_update') refreshContent();
  };
  ws.onclose = function() { setTimeout(connectWS, 2000); };
  ws.onerror = function() { ws.close(); };
}
connectWS();

function refreshContent() {
  if (_contentRefreshTimer) clearTimeout(_contentRefreshTimer);
  _contentRefreshTimer = setTimeout(function() {
    fetch('/api/stats').then(function(r) { return r.json(); }).then(function(s) {
      var newCount = s.articles || 0;
      if (newCount > _lastArticleCount) {
        if (_autoRefreshEnabled) { location.reload(); }
        else {
          var banner = document.getElementById('new-articles-banner');
          if (banner) {
            banner.querySelector('span').textContent = (newCount - _lastArticleCount) + ' new articles';
            banner.style.display = 'flex';
          }
        }
      }
      _lastArticleCount = newCount;
      updateStats();
    }).catch(function() {});
  }, 2000);
}

function updateCrawlUI(state) {
  crawlRunning = state.running || state.continuous;
  var b1 = document.getElementById('btn-crawl-once');
  var b2 = document.getElementById('btn-crawl-continuous');
  var b3 = document.getElementById('btn-crawl-stop');
  if (b1) b1.disabled = crawlRunning;
  if (b2) b2.disabled = crawlRunning;
  if (b3) b3.disabled = !crawlRunning;
  var cs = document.getElementById('crawl-status');
  if (cs) {
    var step = state.current_step || '';
    cs.textContent = step === 'error' ? 'Error' : step === 'done' ? 'Done' : step === 'sleeping' ? 'Sleeping' : step === 'next_tick' ? 'Next tick...' : crawlRunning ? step : 'Idle';
    cs.style.color = step === 'error' ? '#FF6B6B' : crawlRunning ? '#A6F0FF' : '';
    if (state.mode && !crawlRunning) {
      cs.textContent = state.mode;
      cs.style.color = state.mode === 'backward' ? '#FFC107' : '#0BD6AB';
    }
  }
  var modeEl = document.getElementById('crawl-mode-indicator');
  if (modeEl && state.mode) {
    modeEl.textContent = state.mode;
    modeEl.className = 'mode-badge ' + state.mode;
  }
  var isDone = state.current_step === 'done';
  var isError = state.current_step === 'error';
  var showPanel = state.running || state.current_step === 'sleeping' || state.current_step === 'next_tick' || isError || (!isDone && state.current_step !== 'idle');
  var panel = document.getElementById('progress-panel');
  if (panel) panel.style.display = showPanel ? 'block' : 'none';
  var se = document.getElementById('progress-step');
  if (se) { se.textContent = state.current_step || ''; se.style.color = isError ? '#FF6B6B' : ''; }
  var me = document.getElementById('progress-message');
  if (me) me.textContent = state.message || '';
  if (state.stats && Object.keys(state.stats).length) {
    var le = document.getElementById('progress-log');
    if (le) le.innerHTML = JSON.stringify(state.stats, null, 2);
  }
  if (isError) {
    var le = document.getElementById('progress-log');
    if (le) le.innerHTML = '<span style="color:#FF6B6B">' + (state.message || 'Unknown error') + '</span>';
  }
  if (isDone && !state.continuous && panel) { setTimeout(function() { panel.style.display = 'none'; }, 5000); }
}

async function runCrawl(continuous) {
  var feeds = {};
  document.querySelectorAll('#feeds-list .feed-row').forEach(function(row) {
    var s = row.querySelector('.feed-source').value.trim();
    var u = row.querySelector('.feed-url').value.trim();
    if (s && u) feeds[s] = u;
  });
  var seeds = [];
  document.querySelectorAll('#seeds-list .seed-row').forEach(function(row) {
    var u = row.querySelector('.seed-url').value.trim();
    if (u) seeds.push(u);
  });
  var body = {
    feeds: feeds, seeds: seeds,
    max_articles: parseInt(document.getElementById('max-articles').value) || 100,
    max_stories: parseInt(document.getElementById('max-stories').value) || 0,
    continuous: continuous,
    interval: parseInt(document.getElementById('interval').value) || 300,
    llm_enabled: document.getElementById('llm-enabled') ? document.getElementById('llm-enabled').checked : false,
    searxng_enabled: document.getElementById('searxng-enabled') ? document.getElementById('searxng-enabled').checked : true,
    searxng_time_range: document.getElementById('searxng-time-range') ? document.getElementById('searxng-time-range').value : 'day',
    searxng_categories: document.getElementById('searxng-categories') ? document.getElementById('searxng-categories').value : 'news',
    smart_crawl: document.getElementById('smart-crawl') ? document.getElementById('smart-crawl').checked : true,
  };
  try {
    var resp = await fetch('/api/crawl/run', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body) });
    var data = await resp.json();
    if (data.error) alert(data.error);
    else if (continuous) { var p = document.getElementById('progress-panel'); if (p) p.style.display = 'block'; }
    saveCrawlState();
  } catch(e) { alert('Failed: ' + e.message); }
}

async function stopCrawl() {
  try { await fetch('/api/crawl/stop', {method: 'POST'}); } catch(e) {}
}

// Crawler state persistence
var SETTINGS_KEY = 'figtree-news-crawl-settings';
var SETTING_IDS = ['max-articles', 'max-stories', 'interval', 'llm-enabled', 'searxng-enabled', 'searxng-time-range', 'searxng-categories', 'smart-crawl'];

function saveSettings() {
  var settings = {};
  SETTING_IDS.forEach(function(id) {
    var el = document.getElementById(id);
    if (!el) return;
    settings[id] = el.type === 'checkbox' ? el.checked : el.value;
  });
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
}

function loadSettings() {
  try {
    var raw = localStorage.getItem(SETTINGS_KEY);
    if (!raw) return;
    var settings = JSON.parse(raw);
    SETTING_IDS.forEach(function(id) {
      var el = document.getElementById(id);
      if (!el || !(id in settings)) return;
      if (el.type === 'checkbox') el.checked = settings[id];
      else el.value = settings[id];
    });
  } catch(e) {}
}
loadSettings();
SETTING_IDS.forEach(function(id) {
  var el = document.getElementById(id);
  if (el) el.addEventListener('change', saveSettings);
});

async function saveCrawlState() {
  var state = {};
  SETTING_IDS.forEach(function(id) {
    var el = document.getElementById(id);
    if (!el) return;
    state[id] = el.type === 'checkbox' ? el.checked : el.value;
  });
  try { await fetch('/api/crawl/state', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(state) }); } catch(e) {}
}

async function loadCrawlState() {
  try {
    var r = await fetch('/api/crawl/state');
    var state = await r.json();
    for (var key in state) {
      var el = document.getElementById(key);
      if (!el) continue;
      if (el.type === 'checkbox') el.checked = state[key];
      else el.value = state[key];
    }
    var modeEl = document.getElementById('crawl-mode-indicator');
    if (modeEl && state.mode) {
      modeEl.textContent = state.mode;
      modeEl.className = 'mode-badge ' + state.mode;
    }
  } catch(e) {}
}
loadCrawlState();

// Override with server config if available
fetch('/api/config').then(function(r) { return r.json(); }).then(function(cfg) {
  if (cfg.feeds && Object.keys(cfg.feeds).length) {
    document.getElementById('feeds-list').innerHTML = '';
    Object.entries(cfg.feeds).forEach(function(entry) { addFeedRow(entry[0], entry[1]); });
  }
  if (cfg.seeds && cfg.seeds.length) {
    document.getElementById('seeds-list').innerHTML = '';
    cfg.seeds.forEach(function(url) { addSeedRow(url); });
  }
  if (cfg.searxng) {
    var s = cfg.searxng;
    var el;
    el = document.getElementById('searxng-enabled'); if (el) el.checked = s.enabled !== false;
    el = document.getElementById('searxng-time-range'); if (el && s.time_range) el.value = s.time_range;
    el = document.getElementById('searxng-categories'); if (el && s.categories) el.value = s.categories;
  }
}).catch(function() {});

// Feed/Seed rows
function addFeedRow(source, url) {
  var div = document.createElement('div');
  div.className = 'feed-row';
  div.innerHTML = '<input type="text" class="feed-source" placeholder="source" value="' + (source||'') + '"><input type="url" class="feed-url" placeholder="Feed URL" value="' + (url||'') + '"><button class="btn btn-sm btn-danger" onclick="this.parentElement.remove()">&times;</button>';
  document.getElementById('feeds-list').appendChild(div);
}

function addSeedRow(url) {
  var div = document.createElement('div');
  div.className = 'seed-row';
  div.innerHTML = '<input type="url" class="seed-url" placeholder="Seed URL" value="' + (url||'') + '"><button class="btn btn-sm btn-danger" onclick="this.parentElement.remove()">&times;</button>';
  document.getElementById('seeds-list').appendChild(div);
}
