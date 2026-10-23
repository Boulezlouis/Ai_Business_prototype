import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

function App() {
  const [bakken, setBakken] = useState({});
  const [history, setHistory] = useState([]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await fetch('http://localhost:8000/api/bakken');
        const data = await res.json();
        setBakken(data);

        // Voeg data van Bak 07 (Tomaten) toe aan de grafiek geschiedenis
        if (data['07']) {
          const latest = data['07'];
          setHistory(prev => {
            const timeStr = latest.timestamp.split(' ')[1]; // haal alleen de tijdstempel op
            // Voorkom dubbele tijden in de grafiek
            if (prev.length > 0 && prev[prev.length - 1].time === timeStr) return prev;
            
            const newHist = [...prev, { time: timeStr, verkocht_gram: latest.gewicht_afname_gram }];
            return newHist.slice(-15); // Bewaar enkel de laatste 15 metingen voor een mooie live grafiek
          });
        }
      } catch (e) {
        console.error("Kan niet verbinden met de backend", e);
      }
    };
    
    // Haal elke seconde nieuwe data op
    const interval = setInterval(fetchData, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{ padding: '40px', maxWidth: '1200px', margin: '0 auto' }}>
      <header style={{ marginBottom: '40px', borderBottom: '2px solid #ddd', paddingBottom: '20px' }}>
        <h1 style={{ margin: 0, color: '#2c3e50' }}>⚖️ GramIQ Live Dashboard</h1>
        <p style={{ margin: '5px 0 0 0', color: '#7f8c8d' }}>Live overzicht van alle slimme weegschalen in de winkel</p>
      </header>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))', gap: '20px', marginBottom: '40px' }}>
        {Object.keys(bakken).length === 0 ? (
          <div style={{ padding: '20px', backgroundColor: '#fff', borderRadius: '8px', borderLeft: '5px solid #f39c12' }}>
            Wachten op data... Zorg dat de backend en simulator draaien.
          </div>
        ) : (
          Object.entries(bakken).map(([id, info]) => (
            <div key={id} style={{ 
              backgroundColor: '#fff', 
              padding: '20px', 
              borderRadius: '10px', 
              boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
              borderTop: '5px solid #27ae60'
            }}>
              <h2 style={{ margin: '0 0 15px 0', fontSize: '1.2rem' }}>Bak {id}</h2>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px' }}>
                <span style={{ color: '#7f8c8d' }}>Product:</span>
                <strong>{info.product}</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px' }}>
                <span style={{ color: '#7f8c8d' }}>Laatste weging:</span>
                <strong style={{ color: '#e74c3c' }}>-{info.gewicht_afname_gram}g</strong>
              </div>
              <div style={{ fontSize: '0.8rem', color: '#bdc3c7', marginTop: '15px', textAlign: 'right' }}>
                {info.timestamp}
              </div>
            </div>
          ))
        )}
      </div>
      
      <div style={{ backgroundColor: '#fff', padding: '30px', borderRadius: '10px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)' }}>
        <h2 style={{ margin: '0 0 20px 0', color: '#2c3e50' }}>📈 Live Verkoop Tomaten (Bak 07)</h2>
        <div style={{ width: '100%', height: 400 }}>
          <ResponsiveContainer>
            <LineChart data={history}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="verkocht_gram" 
                stroke="#e74c3c" 
                name="Gewicht Afname (gram)" 
                strokeWidth={3} 
                dot={{ r: 6 }} 
                activeDot={{ r: 8 }} 
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

export default App;
