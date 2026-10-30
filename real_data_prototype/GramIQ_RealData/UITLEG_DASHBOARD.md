# Het dashboard uitgelegd

Uitleg bij elk onderdeel van het dashboard (http://localhost:3000), om het prototype
te kunnen toelichten in een presentatie of demo.

## In één zin

> We spelen echte verkopen van een supermarkt opnieuw af, en simuleren wat een
> weegschaal onder elke groentebak daarbij zou meten.

## Wat is echt en wat is gesimuleerd?

| Deel | Echt of gesimuleerd |
|---|---|
| Wanneer een klant iets koopt (tijdstip) | **Echt**, uit de kassadata |
| Hoeveel kg een klant koopt | **Echt**, uit de kassadata |
| Drukte per dag, weekend, seizoen | **Echt** |
| Inkoopprijs en verlies per product | **Echt** |
| De AI-modellen | **Getraind en getest op echte verkopen** (training 2020–2022, deze week in juni 2023 nooit gezien) |
| De weegschaalmeting zelf (ruis, hand in de bak, iemand die leunt) | Gesimuleerd |
| Personeel dat bijvult | Gesimuleerd |
| Grootte van de bak, gewicht per stuk | Aangenomen |
| De tijd | Versneld: standaard vanaf 1 juni 2023, 1 echte seconde = 1 minuut in de winkel |

Voorbeeld: om 10:14:32 kocht een echte klant 0,41 kg broccoli. In de simulator neemt
een klant een paar minuten eerder 0,41 kg uit bak 01, en het gewicht daalt met 0,41 kg
plus wat sensorruis.

**Klopt het dat je het nieuwe prototype ziet?** De titel zegt "· echte verkoopdata", de bakken
heten 01 Broccoli, 02 Lotuswortel, 03 Aubergine en 04 Groene paprika, en de datum is
2023. Zie je tomaten, appels of bananen, dan draait het oude prototype nog.

---

## 1. De bakkaarten (bovenaan)

Eén kaart per weegschaal. Klik op een kaart om haar grafiek te tonen.

| Onderdeel | Uitleg |
|---|---|
| **Bak 01 · Broccoli** | Welke weegschaal, met welk product |
| **Status (kleur + tekst)** | Zie tabel hieronder |
| **Vulbalk** | Hoe vol de bak is, in % van een volle bak |
| **Gewicht** | Gefilterd gewicht in de bak (kg) en % vol. Gefilterd = korte pieken (hand in de bak, iemand die leunt) zijn eruit gehaald met een mediaanfilter over de laatste 7 metingen |
| **Geschat aantal** | Gewicht ÷ gemiddeld gewicht van één stuk (bv. broccoli 400 g). Een schatting |
| **Verbruik (30 min)** | Hoeveel gram per minuut eruit gaat, gemiddeld over het laatste halfuur. Bijvullen telt niet mee |
| **AI-voorspelling: leeg om** | Het tijdstip waarop de bak leeg is **als niemand bijvult**, met tussen haakjes hoeveel minuten dat nog is |
| **marge 13:55 – 14:50** | In ±80% van de gevallen valt het echte moment binnen deze marge |
| **volgende openingsdag** | De bak raakt vandaag niet meer leeg. De voorspelling telt alleen openingsuren (9u–22u), dus het moment valt morgen |
| **Nog te weinig data (min. 10 min)** | De AI heeft minstens 10 minuten metingen nodig, bv. net na het opstarten |
| **ruwe meting … g** (grijs, onderaan) | Het onbewerkte getal van de weegschaal, met het tijdstip van de meting |

### Status

| Status | Kleur | Wanneer |
|---|---|---|
| **In orde** | groen | Geen van de regels hieronder |
| **Binnenkort bijvullen** | oranje | Leeg binnen 45 min, of minder dan 25% vol |
| **Nu bijvullen!** | rood | Leeg binnen 15 min, of minder dan 10% vol |
| **LEEG** | donkerrood | Minder dan 3% vol |

---

## 2. De grafiek (midden)

Toont de geselecteerde bak over de laatste 4 uur (winkeltijd), één punt per minuut.

| Lijn | Uitleg |
|---|---|
| **Blauwe trap (linkeras, kg)** | Gefilterd gewicht in de bak. Dalende treden = klanten die iets nemen; een sprong omhoog = personeel dat bijvult |
| **Oranje stippellijn (rechteras, minuten)** | De AI-voorspelling "leeg over … min" op dat moment. Daalt normaal mee met de tijd; springt omhoog na bijvullen |

---

## 3. Het bestelvoorstel (onderaan)

Beantwoordt: **hoeveel kg moet de winkel vandaag bij de groothandel bestellen?**
Berekend zoals om 4u 's ochtends: enkel met de verkopen tot en met gisteren en de
inkoopprijs van vandaag. Voorbeeld: broccoli op 1 juni 2023.

| Kolom | Broccoli | Uitleg |
|---|---|---|
| **Verwachte verkoop** | 11,5 kg | Beste schatting voor vandaag: de helft van de dagen meer, de helft minder |
| **Verlies** | 9,3% | Deel dat normaal bederft of weggegooid wordt (uit de dataset) |
| **Bestellen** | 18,0 kg | Het advies, zie hieronder |
| **Inkoopprijs** | ¥10,38/kg | Groothandelsprijs van die dag (Chinese yuan: de data komt uit een Chinese supermarkt) |
| **Kost** | ¥186,67 | Bestellen × inkoopprijs |
| **Verkocht tot 18:29** | 11,9 kg | Wat er vandaag tot nu **echt** verkocht is, zie verder |
| **Totaal inkoop** | | Som van de kosten van de vier producten |

### Hoe "Bestellen" berekend wordt

1. **Veilige hoeveelheid**: de AI voorspelt ook een hoeveelheid die op **80% van de dagen**
   volstaat (broccoli: 16,3 kg). Waarom niet 11,5 kg? Dan heb je de helft van de dagen te weinig.
2. **Verlies erbij**: 16,3 ÷ (1 − 0,093) = **18,0 kg**.

Waarom 80%? Een afweging tussen een lege bak en weggooien. Aan te passen in
`ai/bestelmodel.py` (`KWANTIEL = 0.8`).

Het model kijkt naar: verkoop van gisteren, eergisteren en dezelfde dag vorige week;
gemiddelde van de laatste 7, 14 en 28 dagen; dezelfde weekdag in de laatste 4 weken;
maand; kortingen; inkoopprijs van vandaag t.o.v. normaal; hoe vaak het product de laatste
week in de verkoop was.

### "Verkocht tot …" en het balkje

De enige kolom die **geen voorspelling** is. Elke 5 seconden telt de backend alle echte
verkopen van dat product op, van 00:00 tot het huidige tijdstip van de simulatie (retours
niet meegerekend). Het getal komt rechtstreeks uit de kassadata, niet van de weegschaal.

- **Balkje** = verkocht tot nu ÷ bestelling. **Groen** = nog onder de bestelling, **rood** = meer
  verkocht dan besteld (het product zou op geraakt zijn).
- **Tijdens de dag**: rond 15u ongeveer 40–50% is normaal.
- **Om 22u**: het eindtotaal; vergelijk met *Verwachte verkoop* om te zien hoe goed de voorspelling was.

Om te weten:
- **Weegschaal vs kassa**: in de simulatie neemt een klant het product 2–15 minuten vóór het
  afrekenen; de weegschaal ziet een aankoop dus vroeger dan de kassa.
- **Lege gesimuleerde bak**: vult het gesimuleerde personeel te laat bij, dan vindt de klant niets
  in de bak, maar zijn echte aankoop telt wel mee in "Verkocht tot".
- **Verlies**: het balkje vergelijkt met de bestelling inclusief verlies (18,0 kg). Strikt genomen
  is er na bederf maar 16,3 kg te verkopen.

### Getest

De regel onder de tabel: getest op januari–juni 2023 (57 producten, nooit gezien tijdens het
trainen). Gemiddeld **4,7 kg** fout per product per dag, tegenover 6,6 kg voor "zelfde dag vorige
week". De bestelling volstond op **78% van de dagen**.

---

## 4. De regel helemaal onderaan

"AI-model geladen · gemiddelde fout op jan–jun 2023: 80 min (simpele berekening: 123 min)".
Dat is de gemiddelde fout van "wanneer is de bak leeg" over **alle** momenten van de dag.

Belangrijker cijfer voor een presentatie: bij een bak die **minder dan 25% vol** is (het moment
om te beslissen over bijvullen) zit de AI **50 min** naast, de simpele berekening **81 min**:
39% minder fout. Figuur: `resultaat_ai_vs_simpel.png`.

"50 minuten fout" klinkt veel: het gaat om voorspellingen tot uren vooruit, en echte klanten
komen in golven. De AI is wel goed gekalibreerd: zegt hij "leeg binnen 30–60 min", dan was de
bak in het midden van die gevallen na 51 minuten leeg.

---

## Beperkingen (als iemand ernaar vraagt)

- **De weegschaal is nog gesimuleerd.** Volgende stap: testen met echte weegschalen (issue #22).
- **Geen restvoorraad** in het bestelvoorstel: het model doet alsof elke dag met een lege voorraad
  begint. Met echte weegschalen weet je 's avonds precies wat er nog ligt.
- **Eén winkel**, en een Chinese supermarkt. "Waarom Chinese data?": het is de enige publieke
  dataset die we vonden met elke verkoop apart, met tijdstip en gewicht. De methode werkt met de
  kassadata van elke winkel.
- **Geen feestdagen of promoties op voorhand**: alleen het effect van vroegere kortingen.
