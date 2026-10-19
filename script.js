// ===== SVM (SMO đơn giản) =====
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const W = canvas.width, H = canvas.height;
let points = []; // {x,y,label} với x,y trong [0,1]
let model = null;

const $ = id => document.getElementById(id);
const getC = () => Math.pow(10, +$('C').value);
const getGamma = () => Math.pow(10, +$('gamma').value);

function kernelFn(type, gamma) {
  if (type === 'rbf') {
    return (a, b) => Math.exp(-gamma * ((a[0]-b[0])**2 + (a[1]-b[1])**2));
  }
  return (a, b) => a[0]*b[0] + a[1]*b[1];
}

function trainSVM(pts, C, K) {
  const n = pts.length;
  const X = pts.map(p => [p.x, p.y]);
  const y = pts.map(p => p.label);
  const a = new Array(n).fill(0);
  let b = 0;
  const f = i => { let s = b; for (let k = 0; k < n; k++) if (a[k]) s += a[k]*y[k]*K(X[k], X[i]); return s; };
  let passes = 0, iter = 0;
  while (passes < 8 && iter++ < 300) {
    let changed = 0;
    for (let i = 0; i < n; i++) {
      const Ei = f(i) - y[i];
      if ((y[i]*Ei < -1e-3 && a[i] < C) || (y[i]*Ei > 1e-3 && a[i] > 0)) {
        let j = Math.floor(Math.random()*(n-1)); if (j >= i) j++;
        const Ej = f(j) - y[j];
        const ai = a[i], aj = a[j];
        let L, Hh;
        if (y[i] !== y[j]) { L = Math.max(0, aj-ai); Hh = Math.min(C, C+aj-ai); }
        else { L = Math.max(0, ai+aj-C); Hh = Math.min(C, ai+aj); }
        if (L === Hh) continue;
        const eta = 2*K(X[i],X[j]) - K(X[i],X[i]) - K(X[j],X[j]);
        if (eta >= 0) continue;
        a[j] = Math.min(Hh, Math.max(L, aj - y[j]*(Ei-Ej)/eta));
        if (Math.abs(a[j]-aj) < 1e-5) continue;
        a[i] = ai + y[i]*y[j]*(aj - a[j]);
        const b1 = b - Ei - y[i]*(a[i]-ai)*K(X[i],X[i]) - y[j]*(a[j]-aj)*K(X[i],X[j]);
        const b2 = b - Ej - y[i]*(a[i]-ai)*K(X[i],X[j]) - y[j]*(a[j]-aj)*K(X[j],X[j]);
        b = (a[i] > 0 && a[i] < C) ? b1 : (a[j] > 0 && a[j] < C) ? b2 : (b1+b2)/2;
        changed++;
      }
    }
    passes = changed === 0 ? passes + 1 : 0;
  }
  const sv = [];
  for (let i = 0; i < n; i++) if (a[i] > 1e-5) sv.push(i);
  return { X, y, a, b, sv, K, predict: x => { let s = b; for (const k of sv) s += a[k]*y[k]*K(X[k], x); return s; } };
}

// ===== Vẽ =====
function draw() {
  ctx.clearRect(0, 0, W, H);
  const step = 5;
  if (model && model.sv.length) {
    for (let px = 0; px < W; px += step) {
      for (let py = 0; py < H; py += step) {
        const v = model.predict([(px+step/2)/W, (py+step/2)/H]);
        const t = Math.min(1, Math.abs(v)/2) * 0.28;
        ctx.fillStyle = v > 0 ? `rgba(37,99,235,${t})` : `rgba(220,38,38,${t})`;
        ctx.fillRect(px, py, step, step);
        const tol = 0.07;
        if (Math.abs(v) < tol) { ctx.fillStyle = '#111'; ctx.fillRect(px, py, step, step); }
        else if (Math.abs(Math.abs(v)-1) < tol) { ctx.fillStyle = 'rgba(0,0,0,.35)'; ctx.fillRect(px, py, step, step); }
      }
    }
  }
  points.forEach((p, i) => {
    const isSV = model && model.sv.includes(i);
    ctx.beginPath();
    ctx.arc(p.x*W, p.y*H, 7, 0, Math.PI*2);
    ctx.fillStyle = p.label > 0 ? '#2563eb' : '#dc2626';
    ctx.fill();
    ctx.lineWidth = isSV ? 3 : 1;
    ctx.strokeStyle = isSV ? '#111' : '#fff';
    ctx.stroke();
  });
  $('info').textContent = `Số điểm: ${points.length} | Support vectors: ${model ? model.sv.length : 0} (viền đen)`;
}

function retrain() {
  const hasBoth = points.some(p => p.label > 0) && points.some(p => p.label < 0);
  model = hasBoth ? trainSVM(points, getC(), kernelFn($('kernel').value, getGamma())) : null;
  $('cVal').textContent = getC().toFixed(2);
  $('gVal').textContent = getGamma().toFixed(2);
  draw();
}

// ===== Sự kiện =====
function addPoint(e, label) {
  const r = canvas.getBoundingClientRect();
  points.push({ x: (e.clientX - r.left)/r.width, y: (e.clientY - r.top)/r.height, label });
  retrain();
}
canvas.addEventListener('click', e => addPoint(e, e.shiftKey ? -1 : 1));
canvas.addEventListener('contextmenu', e => { e.preventDefault(); addPoint(e, -1); });
['C', 'gamma', 'kernel'].forEach(id => $(id).addEventListener('input', () => {
  $('gammaWrap').hidden = $('kernel').value !== 'rbf';
  retrain();
}));
$('clear').onclick = () => { points = []; retrain(); };
$('sample').onclick = () => {
  points = [];
  const g = () => (Math.random()+Math.random()+Math.random()-1.5)*0.12;
  for (let i = 0; i < 25; i++) {
    points.push({ x: 0.28+g(), y: 0.35+g(), label: 1 });
    points.push({ x: 0.72+g(), y: 0.65+g(), label: -1 });
  }
  retrain();
};

// ===== Nạp CSV từ Colab =====
$('csv').addEventListener('change', e => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    const rows = reader.result.trim().split(/\r?\n/).map(r => r.split(',').map(s => s.trim()));
    const header = rows.shift();
    // Bảng
    let html = '<table><tr>' + header.map(h => `<th>${h}</th>`).join('') + '</tr>';
    rows.slice(0, 100).forEach(r => html += '<tr>' + r.map(c => `<td>${c}</td>`).join('') + '</tr>');
    $('tableWrap').innerHTML = html + '</table>';
    // Chuẩn hóa 2 đặc trưng đầu về [0.05, 0.95]
    const xs = rows.map(r => +r[0]), ys = rows.map(r => +r[1]);
    const norm = (arr, v) => 0.05 + 0.9*(v - Math.min(...arr))/((Math.max(...arr) - Math.min(...arr)) || 1);
    const labels = rows.map(r => r[r.length-1]);
    const first = labels[0];
    points = rows.map((r, i) => ({ x: norm(xs, xs[i]), y: norm(ys, ys[i]), label: labels[i] === first ? 1 : -1 }));
    retrain();
  };
  reader.readAsText(file);
});

$('sample').click();

// ===== Phân loại email: TF-IDF + SVM linear =====
const SPAM = [
"Chúc mừng bạn trúng thưởng 500 triệu đồng, bấm vào link để nhận ngay",
"Giảm giá 90% chỉ hôm nay, mua ngay kẻo hết hàng",
"Vay tiền nhanh không cần thế chấp, giải ngân trong 5 phút",
"Nhận quà miễn phí, ưu đãi có hạn, đăng ký ngay",
"Kiếm 10 triệu mỗi ngày tại nhà, không cần vốn",
"Bạn đã trúng iPhone miễn phí, nhập thông tin để nhận thưởng",
"Khuyến mãi lớn cuối năm, mua một tặng một, bấm link xem ngay",
"Tài khoản ngân hàng của bạn bị khóa, đăng nhập tại link này để xác minh",
"Cơ hội đầu tư lợi nhuận 300% mỗi tháng, tham gia ngay",
"Click vào đây để nhận voucher 1 triệu đồng miễn phí",
"Thuốc giảm cân thần kỳ, giảm 10kg trong 7 ngày, đặt hàng ngay",
"Bạn được chọn nhận khoản vay ưu đãi lãi suất 0%, liên hệ ngay",
"Trúng thưởng xe máy SH, gửi phí nhận thưởng để nhận giải",
"Mua sắm giá sốc, freeship toàn quốc, số lượng có hạn",
"Đăng ký ngay để nhận 5 triệu đồng tiền thưởng chào mừng",
"Cảnh báo tài khoản bị hạn chế, nhấn vào liên kết để khôi phục",
"Xổ số may mắn đã chọn bạn, nhận giải thưởng lớn hôm nay",
"Tuyển cộng tác viên online, thu nhập 15 triệu mỗi tuần, không cần kinh nghiệm",
"Siêu sale 11.11, giảm đến 70% mọi sản phẩm, bấm xem ngay",
"Bạn nhận được quà tặng đặc biệt, bấm vào link để xác nhận"];
const HAM = [
"Mai 9h họp nhóm dự án ở phòng 302, mọi người nhớ đến đúng giờ",
"Gửi anh bản báo cáo tuần này, anh xem giúp em nhé",
"Lịch thi cuối kỳ đã được cập nhật trên cổng thông tin sinh viên",
"Cuối tuần này cả nhà mình đi ăn tối nhé",
"Em nộp bài tập nhóm trước thứ Sáu được không ạ",
"Nhắc lịch khám răng của bạn vào 14h chiều thứ Ba",
"Thầy gửi lại slide bài giảng tuần trước, các em xem trước khi lên lớp",
"Hóa đơn tiền điện tháng này đã được gửi kèm trong file đính kèm",
"Chiều nay họp lúc 3h, nhớ mang theo báo cáo doanh thu",
"Cảm ơn chị đã gửi tài liệu, em sẽ đọc và phản hồi trước ngày mai",
"Đơn hàng của bạn đã được giao thành công, cảm ơn bạn đã mua sắm",
"Anh chị vui lòng xác nhận lịch phỏng vấn vào thứ Tư tuần sau",
"Tối nay em có thể gọi điện trao đổi về kế hoạch đồ án không",
"Biên bản cuộc họp hôm qua em đã gửi trong file đính kèm",
"Nhắc nhở hạn nộp báo cáo thực tập là ngày 20 tháng này",
"Mình gửi bạn địa chỉ quán cà phê, tám giờ sáng mai gặp nhau nhé",
"Phòng đào tạo thông báo lịch nghỉ lễ của trường trong dịp sắp tới",
"Em xin phép nghỉ buổi học ngày mai vì có việc gia đình",
"Báo giá dịch vụ như đã trao đổi, anh xem và góp ý giúp em",
"Chúc mừng sinh nhật bạn, chúc bạn một năm mới nhiều niềm vui"];

function tokens(t) {
  const w = t.toLowerCase().replace(/[^\p{L}\p{N}\s]/gu, ' ').split(/\s+/).filter(s => s.length >= 2);
  const out = [...w];
  for (let i = 0; i < w.length - 1; i++) out.push(w[i] + ' ' + w[i+1]);
  return out;
}
function rng(seed) { return () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function splitData() {
  const r = rng(42), tr = [], te = [];
  [[SPAM, 1], [HAM, -1]].forEach(([arr, lab]) => {
    const idx = arr.map((_, i) => [r(), i]).sort((a, b) => a[0] - b[0]).map(x => x[1]);
    const nTe = Math.round(arr.length * 0.3);
    idx.forEach((i, k) => (k < nTe ? te : tr).push({ text: arr[i], y: lab }));
  });
  return { tr, te };
}
function buildTfidf(docs) {
  const vocab = new Map(), df = [];
  docs.forEach(d => new Set(tokens(d)).forEach(t => {
    if (!vocab.has(t)) { vocab.set(t, vocab.size); df.push(0); }
    df[vocab.get(t)]++;
  }));
  const n = docs.length;
  const idf = df.map(c => Math.log((1 + n) / (1 + c)) + 1);
  const vec = text => {
    const v = new Float64Array(vocab.size), cnt = {};
    tokens(text).forEach(t => cnt[t] = (cnt[t] || 0) + 1);
    for (const t in cnt) if (vocab.has(t)) { const i = vocab.get(t); v[i] = (1 + Math.log(cnt[t])) * idf[i]; }
    const norm = Math.sqrt(v.reduce((s, x) => s + x*x, 0)) || 1;
    return v.map(x => x / norm);
  };
  return { vocab, vec };
}
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]*b[i]; return s; };

function trainLinear(X, y, C) {
  const n = X.length, a = new Array(n).fill(0); let b = 0;
  const G = X.map(u => X.map(v => dot(u, v)));
  const f = i => { let s = b; for (let k = 0; k < n; k++) if (a[k]) s += a[k]*y[k]*G[k][i]; return s; };
  const r = rng(7); let passes = 0, it = 0;
  while (passes < 10 && it++ < 500) {
    let ch = 0;
    for (let i = 0; i < n; i++) {
      const Ei = f(i) - y[i];
      if ((y[i]*Ei < -1e-3 && a[i] < C) || (y[i]*Ei > 1e-3 && a[i] > 0)) {
        let j = Math.floor(r()*(n-1)); if (j >= i) j++;
        const Ej = f(j) - y[j], ai = a[i], aj = a[j];
        const [L, H] = y[i] !== y[j] ? [Math.max(0, aj-ai), Math.min(C, C+aj-ai)] : [Math.max(0, ai+aj-C), Math.min(C, ai+aj)];
        const eta = 2*G[i][j] - G[i][i] - G[j][j];
        if (L === H || eta >= 0) continue;
        a[j] = Math.min(H, Math.max(L, aj - y[j]*(Ei-Ej)/eta));
        if (Math.abs(a[j]-aj) < 1e-5) continue;
        a[i] = ai + y[i]*y[j]*(aj - a[j]);
        const b1 = b - Ei - y[i]*(a[i]-ai)*G[i][i] - y[j]*(a[j]-aj)*G[i][j];
        const b2 = b - Ej - y[i]*(a[i]-ai)*G[i][j] - y[j]*(a[j]-aj)*G[j][j];
        b = (a[i] > 0 && a[i] < C) ? b1 : (a[j] > 0 && a[j] < C) ? b2 : (b1+b2)/2;
        ch++;
      }
    }
    passes = ch === 0 ? passes + 1 : 0;
  }
  const w = new Float64Array(X[0].length);
  a.forEach((ai, k) => { if (ai) for (let d = 0; d < w.length; d++) w[d] += ai*y[k]*X[k][d]; });
  return { w, b, nSV: [a.filter((v, k) => v > 1e-5 && y[k] > 0).length, a.filter((v, k) => v > 1e-5 && y[k] < 0).length] };
}

let mail = null;
function trainMail() {
  const { tr, te } = splitData(), C = +$('mailC').value;
  const T = buildTfidf(tr.map(d => d.text));
  const svm = trainLinear(tr.map(d => T.vec(d.text)), tr.map(d => d.y), C);
  mail = { T, svm };
  let tp = 0, fp = 0, fn = 0, tn = 0;
  te.forEach(d => { const p = dot(svm.w, T.vec(d.text)) + svm.b > 0 ? 1 : -1;
    if (d.y > 0) p > 0 ? tp++ : fn++; else p > 0 ? fp++ : tn++; });
  const pre = tp / ((tp + fp) || 1), rec = tp / ((tp + fn) || 1), f1 = 2*pre*rec / ((pre + rec) || 1);
  const words = [...T.vocab.keys()], idx = [...words.keys()].sort((i, j) => svm.w[j] - svm.w[i]);
  const row = (i) => `${words[i]} (${svm.w[i].toFixed(2)})`;
  $('mailResult').innerHTML = `
    <div class="stats">
      <div class="stat"><b>${((tp+tn)/te.length*100).toFixed(1)}%</b>Accuracy</div>
      <div class="stat"><b>${pre.toFixed(2)}</b>Precision</div>
      <div class="stat"><b>${rec.toFixed(2)}</b>Recall</div>
      <div class="stat"><b>${f1.toFixed(2)}</b>F1</div>
      <div class="stat"><b>${svm.nSV[1]} / ${svm.nSV[0]}</b>SV (hợp lệ / rác)</div>
      <div class="stat"><b>${words.length}</b>Số chiều từ điển</div>
    </div>
    <table class="cm"><tr><th></th><th>Dự đoán: Hợp lệ</th><th>Dự đoán: Thư rác</th></tr>
      <tr><th>Thực tế: Hợp lệ</th><td>${tn}</td><td>${fp}</td></tr>
      <tr><th>Thực tế: Thư rác</th><td>${fn}</td><td>${tp}</td></tr></table>
    <table style="margin-top:12px"><tr><th>Từ đặc trưng thư rác</th><th>Từ đặc trưng thư hợp lệ</th></tr>
      ${[0,1,2,3,4].map(k => `<tr><td>${row(idx[k])}</td><td>${row(idx[idx.length-1-k])}</td></tr>`).join('')}</table>
    <p class="hint">Huấn luyện: ${tr.length} email | Kiểm tra: ${te.length} email</p>`;
  predictMail();
}
function predictMail() {
  if (!mail) return;
  const v = mail.T.vec($('mailInput').value), d = dot(mail.svm.w, v) + mail.svm.b;
  $('mailOut').innerHTML = `Kết quả: <span class="${d > 0 ? 'spam' : 'ham'}">${d > 0 ? 'THƯ RÁC' : 'HỢP LỆ'}</span> (decision = ${d.toFixed(2)})`;
}
$('mailTrain').onclick = trainMail;
$('mailPredict').onclick = predictMail;
trainMail();

// ===== Hiệu ứng giao diện =====
// Hiện dần khi cuộn
const io = new IntersectionObserver(es => es.forEach(e => {
  if (e.isIntersecting) { e.target.classList.add('show'); io.unobserve(e.target); }
}), { threshold: 0.12 });
document.querySelectorAll('.reveal').forEach(el => io.observe(el));

// Thanh tiến trình + menu active
const links = [...document.querySelectorAll('.topnav a')];
const secs = links.map(a => document.querySelector(a.getAttribute('href')));
addEventListener('scroll', () => {
  const h = document.documentElement;
  $('progress').style.width = (h.scrollTop / (h.scrollHeight - h.clientHeight) * 100) + '%';
  let cur = -1;
  secs.forEach((s, i) => { if (s && s.getBoundingClientRect().top < innerHeight * 0.4) cur = i; });
  links.forEach((a, i) => a.classList.toggle('active', i === cur));
}, { passive: true });

// Số chạy lên ở các ô thống kê
function countUp(el) {
  const txt = el.textContent, m = txt.match(/^[\d.]+/);
  if (!m || txt.includes('/')) return;
  const end = parseFloat(m[0]), suffix = txt.slice(m[0].length), dec = (m[0].split('.')[1] || '').length;
  const t0 = performance.now();
  (function tick(t) {
    const p = Math.min(1, (t - t0) / 900), e = 1 - Math.pow(1 - p, 3);
    el.textContent = (end * e).toFixed(dec) + suffix;
    if (p < 1) requestAnimationFrame(tick);
  })(t0);
}
new MutationObserver(() => {
  document.querySelectorAll('#mailResult .stat b').forEach(countUp);
}).observe($('mailResult'), { childList: true });
document.querySelectorAll('#mailResult .stat b').forEach(countUp);
