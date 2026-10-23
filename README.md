Hardware en firmware

Weegschaal kalibreren: ruwe waarde omzetten naar gram (M)
Firmware: gewicht elke X seconden uitlezen en versturen (M)
WiFi-verbinding met automatisch herverbinden (M)
Elke weegschaal een uniek ID geven, gekoppeld aan een bak/product (M)
Tarra-functie: gewicht van de lege bak aftrekken (S)

Communicatie en backend (winkel-PC)
MQTT-broker installeren op de PC (M)
Backend-service die metingen ontvangt en opslaat (M)
Database-structuur: weegschalen, producten, metingen (M)
Gewicht omzetten naar “% vol” en “aantal stuks” per product (M)
Data bufferen als de verbinding wegvalt (S)
Ruis en uitschieters filteren, bv. iemand die op de bak leunt (S)

Kassa-integratie
Verkoopdata importeren uit de kassa (CSV-export als eerste versie) (C)
Verkocht en verdwenen gewicht vergelijken, verschil tonen (verspilling/diefstal) (C)

Dashboard en meldingen
Overzicht van alle bakken met hun huidig niveau (M)
Melding bij een laag niveau, met een instelbare drempel (M)
Grafiek van het niveau doorheen de dag per product (S)
Instellingenpagina: weegschaal koppelen aan product (S)
Meldingen op gsm of smartwatch van het personeel (C)

Analyse en AI
Rapport: piekuren en verkoop per product per dag (S)
Voorspellen wanneer een bak leeg zal zijn (C)
Voorspellen hoeveel er besteld moet worden (C)
Cloud-sync en vergelijking tussen meerdere winkels (W)

Algemeen
Git-repo en projectstructuur opzetten (M)
Testen met 2–3 weegschalen in een testopstelling (M)
Installatiehandleiding voor de winkel (S)
Beveiliging: MQTT met login, dashboard enkel lokaal bereikbaar (S)
