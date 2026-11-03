Risk mot tillväxt i Tres telefonkrediter

Ett case om hur Tre kan godkänna fler kunder utan högre risk, genom hela konjunkturen. Caset kombinerar kreditriskmodellering med en scenarioanalys inom telekom och är avgränsat till tre frågor:
Vad kostar tillväxt i kreditförluster, och var sätts gränsen?

Ger motbud, det vill säga billigare telefon, handpenning eller abonnemang utan telefon, mer än ett nej under samma riskaptit?
Vad händer med kreditrisken när konjunkturen försämras, hur upptäcks det tidigt och vilka beslut bör följa?
Rapporten (PDF) beskriver resultaten och slutsatserna. Det här repot innehåller beräkningskoden bakom dem.

Koden

All beräkningskod ligger i tre_case.py, vilket gör att hela analysen kan följas och köras i en och samma fil. Den är uppdelad i sju delar:

Ekonomin: värdet av en bra respektive dålig kund över den diskonterade kundtiden, kostnaden för räntefri delbetalning och brytpunkten där ja och nej är lika mycket värda.

Data: population, utfall och variabler. Ett lån räknas som dåligt först när mer än 10 procent av en delbetalning är obetald efter 30 dagar, eftersom de flesta sena betalningar annars visade sig vara små restbelopp. Alla variabler bygger endast på information från de 730 dagarna före beslutet.

Modeller: ett scorekort med WoE och logistisk regression som beslutsmodell, och LightGBM med monotona villkor som utmanare. Osäkerheten mäts genom att dra kunder och inte lån, eftersom lån hos samma kund inte är oberoende.

Tillväxt och motbud: hela ansökningsflödet, det vill säga både beviljade och nekade ansökningar, används för att jämföra snitt mot marginal och en ren ja-eller-nej-policy mot en meny av motbud under samma tak för förväntad kreditförlust.

Konjunkturkänslighet: en diskret hasardmodell på Lending Club mot arbetslösheten per delstat, med fasta effekter per kalendermånad och standardfel klustrade per delstat.

Stresstest: stressad PD i log-odds för arbetslöshet och köpkraftschocker, kompletterat med ett omvänt stresstest och Monte Carlo.

Kreditstegen: hur betalningen av de första fakturorna förutsäger risken under de följande månaderna.

Data

Rådatan ingår inte i repot, eftersom Kaggles villkor inte tillåter att den sprids. Den läggs i en mapp som anges med TRE_DATA:
hcdr/: Home Credit Default Risk (Kaggle)
lc/: Lending Club 2007–2018 (Kaggle)
macro/: arbetslöshet per delstat (BLS via Kaggle)
sverige/: arbetslöshet per månad (SCB) och Konjunkturinstitutets prognos från september 2026

Körning

pip install polars pandas numpy scipy scikit-learn statsmodels lightgbm
TRE_DATA=/sökväg/till/data python tre_case.py
Körningen tar omkring fem minuter och ger samma resultat varje gång. Resultaten sparas i outputs/.

Resultatens status

Resultaten bör läsas utifrån vad de bygger på. Observerat är det som är mätt direkt i Home Credits och Lending Clubs data, vilket alltså inte är Tres data. Simulerat är det som kombinerar observerad risk med antaganden, till exempel bidrag, brytpunkter, menyns värde och stresscenarier. Antaget är bland annat marginal, LGD, de nekades risk och acceptansen för motbud. Metoden och strukturen i besluten går därmed att överföra, medan nivåerna behöver valideras på Tres egna data.
