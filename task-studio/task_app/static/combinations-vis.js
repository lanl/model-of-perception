/* Parallel stage view. No chart dependency; the backend remains authoritative. */
(function (global) {
  'use strict';
  const STAGES = ['operation', 'selection', 'aggregation', 'action'];
  const TITLES = ['Operation', 'Selection', 'Aggregation', 'Action'];
  const NS = 'http://www.w3.org/2000/svg';

  function colorFor(id) {
    let hash = 2166136261;
    for (const char of id) hash = Math.imul(hash ^ char.charCodeAt(0), 16777619);
    return `hsl(${(hash >>> 0) % 360} 62% 42%)`;
  }

  function buildLayout(items, options) {
    const columns = STAGES.map((key, index) => ({
      key, title: TITLES[index], x: 140 + index * 300,
      values: Object.keys(options[key]).filter(value => items.some(item => item.stages[key] === value)),
    }));
    const rowCount = Math.max(1, ...columns.map(column => column.values.length));
    const height = Math.max(300, 108 + rowCount * 66);
    const nodes = columns.flatMap(column => column.values.map((value, index) => ({
      key: column.key, value, label: options[column.key][value], x: column.x,
      y: 86 + (column.values.length === 1 ? (rowCount - 1) * 33 : index * (rowCount - 1) * 66 / (column.values.length - 1)),
    })));
    const ordered = [...items].sort((a, b) => a.id.localeCompare(b.id));
    const paths = ordered.map((item, index) => {
      // Slight lane separation lets shared routes remain individually visible.
      const lane = ordered.length < 2 ? 0 : -9 + 18 * index / (ordered.length - 1);
      const points = columns.map(column => {
        const node = nodes.find(node => node.key === column.key && node.value === item.stages[column.key]);
        return [column.x, node.y + 20 + lane];
      });
      let d = `M ${points[0][0]} ${points[0][1]}`;
      for (let i = 1; i < points.length; i++) {
        const [x0, y0] = points[i - 1], [x1, y1] = points[i];
        const middle = (x0 + x1) / 2;
        d += ` C ${middle} ${y0}, ${middle} ${y1}, ${x1} ${y1}`;
      }
      return { ...item, points, d, color: colorFor(item.id),
        label: STAGES.map(key => options[key][item.stages[key]]).join(' → ') };
    });
    return { width: 1180, height, columns, nodes, paths };
  }

  function element(name, attributes = {}, text) {
    const node = document.createElementNS(NS, name);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
    if (text !== undefined) node.textContent = text;
    return node;
  }

  class View {
    constructor({ canvas, inspector, info, toggle, onToggle }) {
      Object.assign(this, { canvas, inspector, info, toggle, onToggle });
      this.selected = ''; this.hovered = ''; this.layout = null;
      inspector.addEventListener('change', () => {
        this.selected = inspector.value; this.hovered = this.selected; this.highlight();
      });
      canvas.addEventListener('pointerleave', () => {
        this.hovered = ''; this.highlight();
      });
      toggle.addEventListener('click', () => {
        if (this.selected) onToggle(this.selected);
      });
    }

    render(items, options) {
      const focusedId = document.activeElement?.dataset?.combinationId;
      this.layout = buildLayout(items, options);
      if (!items.some(item => item.id === this.selected)) this.selected = '';
      if (!items.some(item => item.id === this.hovered)) this.hovered = '';
      this.canvas.replaceChildren(); this.inspector.replaceChildren();
      const placeholder = document.createElement('option');
      placeholder.value = ''; placeholder.textContent = 'Inspect a combination…';
      this.inspector.append(placeholder);
      for (const path of this.layout.paths) {
        const option = document.createElement('option');
        option.value = path.id; option.textContent = `${path.included ? 'Included' : 'Excluded'} · ${path.label}`;
        this.inspector.append(option);
      }
      this.inspector.value = this.selected; this.inspector.disabled = !items.length;
      this.groups = new Map();
      if (!items.length) {
        const empty = document.createElement('p'); empty.className = 'combination-empty';
        empty.textContent = 'Choose compatible stage options above to see their connections.';
        this.canvas.append(empty); this.highlight(); return;
      }
      const { width, height, columns, nodes, paths } = this.layout;
      const svg = element('svg', { viewBox: `0 0 ${width} ${height}`, width, height,
        class: 'combination-svg', role: 'group', 'aria-label': 'Compatible combinations through operation, selection, aggregation, and action' });
      svg.append(element('title', {}, 'Each curve is one combination. Click or press Enter or Space to include or exclude it.'));
      for (const column of columns) {
        svg.append(element('rect', { x: column.x - 122, y: 51, width: 244, height: height - 72,
          rx: 12, fill: '#f5f8f3' }));
        svg.append(element('text', { x: column.x, y: 29, 'text-anchor': 'middle', class: 'stage-title' }, column.title));
      }
      for (const node of nodes) {
        svg.append(element('rect', { x: node.x - 111, y: node.y - 17, width: 222, height: 51,
          rx: 8, class: 'stage-node', 'data-stage': node.key, 'data-value': node.value }));
      }
      this.pathLayer = element('g', { class: 'combination-paths' });
      svg.append(this.pathLayer);
      // Excluded paths sit behind the included paths, but remain reachable via inspector.
      for (const path of [...paths].sort((a, b) => Number(a.included) - Number(b.included))) {
        const group = element('g', { class: `combination-curve${path.included ? '' : ' is-excluded'}`,
          role: 'button', tabindex: 0, 'aria-pressed': path.included,
          'aria-label': `${path.included ? 'Included' : 'Excluded'}: ${path.label}. Toggle inclusion.`,
          'data-combination-id': path.id });
        group.append(element('title', {}, `${path.label} (${path.included ? 'included' : 'excluded'})`));
        group.append(element('path', { d: path.d, stroke: path.color, class: 'curve-line' }));
        // The transparent hit stroke is deliberately wider than the visible curve.
        group.append(element('path', { d: path.d, class: 'curve-hit' }));
        for (const [cx, cy] of path.points) {
          group.append(element('circle', { cx, cy, r: 2.5, fill: path.color, class: 'curve-dot' }));
        }
        const enter = () => { this.hovered = path.id; this.highlight(); };
        const leave = () => { this.hovered = ''; this.highlight(); };
        group.addEventListener('pointerenter', enter); group.addEventListener('pointerleave', leave);
        group.addEventListener('focus', enter); group.addEventListener('blur', leave);
        const activate = () => { this.selected = path.id; this.onToggle(path.id); };
        group.addEventListener('click', activate);
        group.addEventListener('keydown', event => {
          if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); activate(); }
        });
        this.pathLayer.append(group); this.groups.set(path.id, group);
      }
      // Labels are above the connection lanes, with no pointer interception.
      for (const node of nodes) {
        svg.append(element('text', { x: node.x, y: node.y, 'text-anchor': 'middle', class: 'stage-label' }, node.label));
      }
      this.svg = svg; this.canvas.append(svg); this.highlight();
      if (focusedId && this.groups.has(focusedId)) {
        // Restoring DOM focus after a redraw must not revive a cleared hover.
        const hovered = this.hovered;
        this.groups.get(focusedId).focus({ preventScroll: true });
        this.hovered = hovered; this.highlight();
      }
    }

    highlight() {
      const activeId = this.hovered;
      const active = this.layout?.paths.find(path => path.id === activeId);
      for (const [id, group] of this.groups || []) {
        group.classList.toggle('is-highlighted', id === activeId);
        group.classList.toggle('is-muted', Boolean(active) && id !== activeId);
      }
      for (const node of this.canvas.querySelectorAll('.stage-node')) {
        node.classList.toggle('is-highlighted', Boolean(active) && active.stages[node.dataset.stage] === node.dataset.value);
      }
      this.info.textContent = active ? `${active.included ? 'Included' : 'Excluded'} · ${active.label}`
        : 'Hover or focus a curve to trace it. Click a curve to toggle inclusion. Use the selector to inspect overlapping curves.';
      this.info.style.borderLeftColor = active?.color || '#c7d6c9';
      const selected = this.layout?.paths.find(path => path.id === this.selected);
      this.toggle.disabled = !selected;
      this.toggle.textContent = selected?.included ? 'Exclude selected' : 'Include selected';
      this.inspector.value = this.selected;
    }
  }
  global.CombinationViz = { buildLayout, colorFor, View };
})(globalThis);
