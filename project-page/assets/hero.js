(() => {
  const canvas = document.querySelector(".causal-field");
  const hero = document.querySelector(".hero-head");
  if (!canvas || !hero) return;

  const ctx = canvas.getContext("2d", { alpha: true });
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const pointer = { x: 0, y: 0, tx: 0, ty: 0, active: false };

  let width = 1;
  let height = 1;
  let dpr = 1;
  let nodes = [];
  let streams = [];
  let frame = 0;
  let previous = 0;
  let running = true;

  const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
  const mix = (a, b, t) => a + (b - a) * t;

  const cubicPoint = (curve, t) => {
    const u = 1 - t;
    return {
      x: u ** 3 * curve.x0 + 3 * u ** 2 * t * curve.x1 + 3 * u * t ** 2 * curve.x2 + t ** 3 * curve.x3,
      y: u ** 3 * curve.y0 + 3 * u ** 2 * t * curve.y1 + 3 * u * t ** 2 * curve.y2 + t ** 3 * curve.y3,
    };
  };

  const resize = () => {
    const rect = hero.getBoundingClientRect();
    width = Math.max(1, Math.round(rect.width));
    height = Math.max(1, Math.round(rect.height));
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    pointer.x = pointer.tx = width / 2;
    pointer.y = pointer.ty = height / 2;

    const nodeCount = clamp(Math.round((width * height) / 12500), 54, 118);
    nodes = Array.from({ length: nodeCount }, (_, index) => ({
      x: Math.random() * width,
      y: Math.random() * height,
      vx: (Math.random() - 0.5) * 0.085,
      vy: (Math.random() - 0.5) * 0.065,
      radius: 0.65 + Math.random() * 1.25,
      phase: Math.random() * Math.PI * 2,
      family: index % 3,
    }));

    const streamCount = width < 720 ? 4 : 7;
    streams = Array.from({ length: streamCount }, (_, index) => {
      const band = (index + 1) / (streamCount + 1);
      const direction = index % 2 === 0 ? 1 : -1;
      return {
        x0: direction > 0 ? -width * 0.12 : width * 1.12,
        y0: height * (band + (Math.random() - 0.5) * 0.12),
        x1: width * (direction > 0 ? 0.24 : 0.76),
        y1: height * (band - 0.2 + Math.random() * 0.16),
        x2: width * (direction > 0 ? 0.72 : 0.28),
        y2: height * (band + 0.05 + Math.random() * 0.2),
        x3: direction > 0 ? width * 1.12 : -width * 0.12,
        y3: height * (band + (Math.random() - 0.5) * 0.18),
        phase: Math.random(),
        speed: (0.018 + Math.random() * 0.018) * direction,
        hue: index % 2,
      };
    });
  };

  const drawStream = (stream, time) => {
    const sway = Math.sin(time * 0.00018 + stream.phase * 8) * height * 0.018;
    const curve = { ...stream, y1: stream.y1 + sway, y2: stream.y2 - sway };
    const color = stream.hue === 0 ? "15,159,145" : "118,80,206";

    ctx.beginPath();
    ctx.moveTo(curve.x0, curve.y0);
    ctx.bezierCurveTo(curve.x1, curve.y1, curve.x2, curve.y2, curve.x3, curve.y3);
    ctx.strokeStyle = `rgba(${color},0.13)`;
    ctx.lineWidth = 0.8;
    ctx.stroke();

    for (let pulse = 0; pulse < 3; pulse += 1) {
      const t = ((time * stream.speed * 0.001 + stream.phase + pulse / 3) % 1 + 1) % 1;
      const point = cubicPoint(curve, t);
      const halo = ctx.createRadialGradient(point.x, point.y, 0, point.x, point.y, 15);
      halo.addColorStop(0, `rgba(${color},0.7)`);
      halo.addColorStop(0.16, `rgba(${color},0.22)`);
      halo.addColorStop(1, `rgba(${color},0)`);
      ctx.fillStyle = halo;
      ctx.beginPath();
      ctx.arc(point.x, point.y, 15, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = `rgba(${color},0.95)`;
      ctx.beginPath();
      ctx.arc(point.x, point.y, 1.45, 0, Math.PI * 2);
      ctx.fill();
    }
  };

  const draw = (time = 0) => {
    const dt = Math.min(32, time - previous || 16.7);
    previous = time;
    pointer.x = mix(pointer.x, pointer.tx, 0.065);
    pointer.y = mix(pointer.y, pointer.ty, 0.065);
    ctx.clearRect(0, 0, width, height);

    streams.forEach((stream) => drawStream(stream, time));

    for (let i = 0; i < nodes.length; i += 1) {
      const node = nodes[i];
      node.x += node.vx * dt;
      node.y += node.vy * dt;
      if (node.x < -30) node.x = width + 30;
      if (node.x > width + 30) node.x = -30;
      if (node.y < -30) node.y = height + 30;
      if (node.y > height + 30) node.y = -30;

      const dx = pointer.x - node.x;
      const dy = pointer.y - node.y;
      const distance = Math.hypot(dx, dy);
      const influence = pointer.active ? clamp(1 - distance / 230, 0, 1) : 0;
      const px = node.x - dx * influence * 0.035;
      const py = node.y - dy * influence * 0.035;
      const pulse = 0.72 + Math.sin(time * 0.0014 + node.phase) * 0.28;
      const colors = ["15,159,145", "35,104,216", "118,80,206"];
      const color = colors[node.family];

      for (let j = i + 1; j < nodes.length; j += 1) {
        const other = nodes[j];
        const edgeDistance = Math.hypot(node.x - other.x, node.y - other.y);
        if (edgeDistance > 132) continue;
        const alpha = (1 - edgeDistance / 132) * 0.105;
        ctx.strokeStyle = `rgba(54,104,184,${alpha * 1.3})`;
        ctx.lineWidth = 0.55;
        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.lineTo(other.x, other.y);
        ctx.stroke();
      }

      if (influence > 0.05) {
        ctx.strokeStyle = `rgba(${color},${influence * 0.32})`;
        ctx.lineWidth = 0.8;
        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.lineTo(pointer.x, pointer.y);
        ctx.stroke();
      }

      ctx.fillStyle = `rgba(${color},${0.25 + pulse * 0.36})`;
      ctx.beginPath();
      ctx.arc(px, py, node.radius + influence * 1.5, 0, Math.PI * 2);
      ctx.fill();
    }

    if (pointer.active) {
      const halo = ctx.createRadialGradient(pointer.x, pointer.y, 0, pointer.x, pointer.y, 160);
      halo.addColorStop(0, "rgba(35,104,216,0.075)");
      halo.addColorStop(1, "rgba(35,104,216,0)");
      ctx.fillStyle = halo;
      ctx.beginPath();
      ctx.arc(pointer.x, pointer.y, 160, 0, Math.PI * 2);
      ctx.fill();
    }

    frame = running && !reducedMotion ? requestAnimationFrame(draw) : 0;
  };

  const updatePointer = (event) => {
    const rect = hero.getBoundingClientRect();
    pointer.tx = event.clientX - rect.left;
    pointer.ty = event.clientY - rect.top;
    pointer.active = true;
  };

  resize();
  draw();

  hero.addEventListener("pointermove", updatePointer, { passive: true });
  hero.addEventListener("pointerleave", () => { pointer.active = false; });
  window.addEventListener("resize", () => {
    cancelAnimationFrame(frame);
    resize();
    previous = 0;
    draw();
  });

  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver((entries) => {
      running = entries.some((entry) => entry.isIntersecting);
      cancelAnimationFrame(frame);
      if (running && !reducedMotion) {
        previous = 0;
        frame = requestAnimationFrame(draw);
      }
    });
    observer.observe(hero);
  }
})();
