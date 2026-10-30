import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const API = 'http://localhost:8000';

const STATUS = {
  ok:      { kleur: '#27ae60', label: 'In orde' },
  laag:    { kleur: '#f39c12', label: 'Binnenkort bijvullen' },
  kritiek: { kleur: '#e74c3c', label: 'Nu bijvullen!' },
  leeg:    { kleur: '#7f1d1d', label: 'LEEG' },
};

function Rij({ label, children }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
      <span style={{ color: '#7f8c8d' }}>{label}</span>
      <strong>{children}</strong>
    </div>
  );
}

function BakKaart({ id, info, geselecteerd, onClick }) {
  const s = STATUS[info.status] || STATUS.ok;
  const v = info.voorspelling;
  const pct = Math.min(100, Math.max(0, info.pct_vol));
  return (
    <div onClick={onClick} style={{
      backgroundColor: '#fff', padding: '20px', borderRadius: '10px', cursor: 'pointer',
      boxShadow: geselecteerd ? `0 0 0 3px ${s.kleur}` : '0 4px 6px rgba(0,0,0,0.1)',
      borderTop: `5px solid ${s.kleur}`,
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
        <h2 style={{ margin: '0 0 4px 0', fontSize: '1.2rem' }}>Bak {id} · {info.product}</h2>
        <span style={{ color: s.kleur, fontWeight: 700, fontSize: '0.85rem' }}>{s.label}</span>
      </div>

      {/* vulniveau */}
      <div style={{ margin: '12px 0 14px', background: '#ecf0f1', borderRadius: '6px', height: '14px', overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: s.kleur, transition: 'width 0.5s' }} />
      </div>

      <Rij label="Gewicht">{(info.gewicht_gram / 1000).toFixed(2)} kg ({info.pct_vol.toFixed(0)}%)</Rij>
      <Rij label="Geschat aantal">{info.stuks} stuks</Rij>
      {v && <Rij label="Verbruik (30 min)">{v.verbruik_g_per_min} g/min</Rij>}

      <div style={{ marginTop: '14px', padding: '12px', background: '#f8f9fa', borderRadius: '8px' }}>
        <div style={{ fontSize: '0.8rem', color: '#7f8c8d', marginBottom: '4px' }}>🤖 AI-voorspelling: leeg om</div>
        {!v ? (
          <div style={{ color: '#95a5a6' }}>Nog te weinig data (min. 10 min)…</div>
        ) : info.status === 'leeg' ? (
          <div style={{ fontSize: '1.4rem', fontWeight: 700, color: s.kleur }}>Nu leeg</div>
        ) : (
          <>
            <div style={{ fontSize: '1.4rem', fontWeight: 700, color: s.kleur }}>
              {v.leeg_om.split(' ')[1]}
              <span style={{ fontSize: '0.9rem', fontWeight: 400, color: '#555' }}> (over {Math.round(v.minuten)} min)</span>
            </div>
            <div style={{ fontSize: '0.8rem', color: '#7f8c8d' }}>
              marge {v.leeg_vroegst} – {v.leeg_laatst}
              {v.leeg_om.split(' ')[0] !== info.timestamp.split(' ')[0] && ' · volgende openingsdag'}
            </div>
          </>
        )}
      </div>

      <div style={{ fontSize: '0.75rem', color: '#bdc3c7', marginTop: '12px', textAlign: 'right' }}>
        ruwe meting {info.gewicht_ruw_gram} g · {info.timestamp}
      </div>
    </div>
  );
}

function BestelPaneel({ data }) {
  if (!data) return null;
  if (!data.beschikbaar) {
    return (
      <div style={{ marginTop: '20px', color: '#7f8c8d', fontSize: '0.85rem' }}>
        ⚠️ Geen bestelmodel geladen. Train het met <code>cd ai &amp;&amp; python bestelmodel.py</code>.
      </div>
    );
  }
  const rijen = Object.entries(data.bakken || {});
  const totaal = rijen.reduce((s, [, b]) => s + (b.kost || 0), 0);
  return (
    <div style={{ marginTop: '30px', backgroundColor: '#fff', padding: '30px', borderRadius: '10px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)' }}>
      <h2 style={{ margin: '0 0 6px 0', color: '#2c3e50' }}>🚚 Bestelvoorstel {data.datum ? `voor ${data.datum}` : ''}</h2>
      <p style={{ margin: '0 0 20px 0', color: '#7f8c8d', fontSize: '0.9rem' }}>
        Berekend om 4u 's ochtends met de verkopen tot en met gisteren. Bestelling = hoeveelheid die op
        {' '}{Math.round((data.kwantiel || 0.8) * 100)}% van de dagen volstaat, plus het normale verlies (bederf).
      </p>
      {rijen.length === 0 ? (
        <div style={{ color: '#95a5a6' }}>Wachten op metingen…</div>
      ) : (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.95rem' }}>
            <thead>
              <tr style={{ textAlign: 'right', color: '#7f8c8d', borderBottom: '2px solid #ecf0f1' }}>
                <th style={{ textAlign: 'left', padding: '8px' }}>Product</th>
                <th style={{ padding: '8px' }}>Verwachte verkoop</th>
                <th style={{ padding: '8px' }}>Verlies</th>
                <th style={{ padding: '8px' }}>Bestellen</th>
                <th style={{ padding: '8px' }}>Inkoopprijs</th>
                <th style={{ padding: '8px' }}>Kost</th>
                <th style={{ padding: '8px', minWidth: '180px' }}>Verkocht tot {data.tijd}</th>
              </tr>
            </thead>
            <tbody>
              {rijen.map(([id, b]) => {
                const pct = Math.min(100, (100 * b.verkocht_tot_nu_kg) / Math.max(b.bestel_kg, 0.1));
                return (
                  <tr key={id} style={{ textAlign: 'right', borderBottom: '1px solid #ecf0f1' }}>
                    <td style={{ textAlign: 'left', padding: '8px' }}><strong>Bak {id}</strong> · {b.product}</td>
                    <td style={{ padding: '8px' }}>{b.verwacht_kg} kg</td>
                    <td style={{ padding: '8px' }}>{b.verlies_pct}%</td>
                    <td style={{ padding: '8px', fontWeight: 700, color: '#2980b9' }}>{b.bestel_kg} kg</td>
                    <td style={{ padding: '8px' }}>{b.inkoopprijs_per_kg != null ? `¥${b.inkoopprijs_per_kg}/kg` : '–'}</td>
                    <td style={{ padding: '8px' }}>{b.kost != null ? `¥${b.kost.toFixed(2)}` : '–'}</td>
                    <td style={{ padding: '8px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', justifyContent: 'flex-end' }}>
                        <div style={{ flex: 1, background: '#ecf0f1', borderRadius: '4px', height: '8px', overflow: 'hidden' }}>
                          <div style={{ width: `${pct}%`, height: '100%', background: pct >= 100 ? '#e74c3c' : '#27ae60' }} />
                        </div>
                        <span>{b.verkocht_tot_nu_kg} kg</span>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
            <tfoot>
              <tr style={{ textAlign: 'right', color: '#2c3e50' }}>
                <td colSpan={5} style={{ padding: '8px' }}>Totaal inkoop</td>
                <td style={{ padding: '8px', fontWeight: 700 }}>¥{totaal.toFixed(2)}</td>
                <td />
              </tr>
            </tfoot>
          </table>
        </div>
      )}
      {data.test && data.test.MAE_model_kg != null && (
        <div style={{ marginTop: '14px', color: '#7f8c8d', fontSize: '0.85rem' }}>
          Getest op jan–jun 2023 (57 producten): gemiddelde fout <strong>{data.test.MAE_model_kg.toFixed(1)} kg</strong> per product per dag
          {' '}(‘zelfde dag vorige week’: {data.test.MAE_vorige_week_kg.toFixed(1)} kg) · bestelling volstaat op
          {' '}{Math.round(100 * data.test.genoeg_met_bestelling)}% van de dagen.
        </div>
      )}
    </div>
  );
}

function App() {
  const [bakken, setBakken] = useState({});
  const [geselecteerd, setGeselecteerd] = useState('01');
  const [history, setHistory] = useState([]);
  const [model, setModel] = useState(null);
  const [bestelling, setBestelling] = useState(null);

  useEffect(() => {
    const haal = () => fetch(`${API}/api/bestelling`).then(r => r.json()).then(setBestelling).catch(() => {});
    haal();
    const i = setInterval(haal, 5000);
    return () => clearInterval(i);
  }, []);

  useEffect(() => {
    fetch(`${API}/api/model`).then(r => r.json()).then(setModel).catch(() => {});
  }, []);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await fetch(`${API}/api/bakken`);
        setBakken(await res.json());
        const h = await fetch(`${API}/api/historiek/${geselecteerd}`);
        if (h.ok) setHistory(await h.json());
      } catch (e) {
        console.error('Kan niet verbinden met de backend', e);
      }
    };
    fetchData();
    const interval = setInterval(fetchData, 1000);
    return () => clearInterval(interval);
  }, [geselecteerd]);

  const sel = bakken[geselecteerd];
  const test = model?.rapport?.['Test (jan-jun 2023)'];

  return (
    <div style={{ padding: '40px', maxWidth: '1200px', margin: '0 auto' }}>
      <header style={{ marginBottom: '40px', borderBottom: '2px solid #ddd', paddingBottom: '20px' }}>
        <h1 style={{ margin: 0, color: '#2c3e50' }}>⚖️ GramIQ Live Dashboard · echte verkoopdata</h1>
        <p style={{ margin: '5px 0 0 0', color: '#7f8c8d' }}>
          Weegschalen onder 4 groentebakken, gevoed met echte verkopen van een supermarkt (2020–2023) ·
          AI-voorspelling wanneer elke bak leeg is · bestelvoorstel per dag
        </p>
      </header>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '20px', marginBottom: '40px' }}>
        {Object.keys(bakken).length === 0 ? (
          <div style={{ padding: '20px', backgroundColor: '#fff', borderRadius: '8px', borderLeft: '5px solid #f39c12' }}>
            Wachten op data... Zorg dat de backend en simulator draaien.
          </div>
        ) : (
          Object.entries(bakken).map(([id, info]) => (
            <BakKaart key={id} id={id} info={info} geselecteerd={id === geselecteerd} onClick={() => setGeselecteerd(id)} />
          ))
        )}
      </div>

      <div style={{ backgroundColor: '#fff', padding: '30px', borderRadius: '10px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)' }}>
        <h2 style={{ margin: '0 0 6px 0', color: '#2c3e50' }}>
          📈 Gewicht en voorspelling – Bak {geselecteerd}{sel ? ` (${sel.product})` : ''}
        </h2>
        <p style={{ margin: '0 0 20px 0', color: '#7f8c8d', fontSize: '0.9rem' }}>
          Gefilterd gewicht per minuut (laatste 4 uur simulatietijd). Klik op een bak om te wisselen.
        </p>
        <div style={{ width: '100%', height: 400 }}>
          <ResponsiveContainer>
            <LineChart data={history}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="tijd" minTickGap={30} />
              <YAxis yAxisId="g" tickFormatter={v => `${(v / 1000).toFixed(0)} kg`} />
              <YAxis yAxisId="m" orientation="right" tickFormatter={v => `${v} min`} />
              <Tooltip />
              <Legend />
              <Line yAxisId="g" type="stepAfter" dataKey="gewicht" stroke="#2980b9" name="Gewicht (gram)" strokeWidth={2} dot={false} isAnimationActive={false} />
              <Line yAxisId="m" type="monotone" dataKey="voorspelling_min" stroke="#e67e22" name="Voorspeld: leeg over (min)" strokeWidth={2} dot={false} strokeDasharray="5 3" isAnimationActive={false} connectNulls />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <BestelPaneel data={bestelling} />

      {model && (
        <div style={{ marginTop: '20px', color: '#7f8c8d', fontSize: '0.85rem' }}>
          {model.geladen && test ? (
            <>🤖 AI-model geladen · gemiddelde fout op jan–jun 2023 (nooit gezien tijdens training): <strong>{test.MAE_model_min.toFixed(1)} min</strong>
              {' '}(simpele berekening gewicht ÷ verbruik: {test.MAE_baseline_min.toFixed(1)} min)</>
          ) : (
            <>⚠️ Geen AI-model geladen – backend gebruikt de simpele berekening. Train het model in de map <code>ai/</code>.</>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
