/* A small, real, aligned English-Kannada phrasebook corpus (travel phrases for a visitor in Karnataka). */
const ITEMS = [["bus stand", "ಬಸ್ ನಿಲ್ದಾಣ", "ಬಸ್ ನಿಲ್ದಾಣಕ್ಕೆ"], ["railway station", "ರೈಲ್ವೆ ನಿಲ್ದಾಣ", "ರೈಲ್ವೆ ನಿಲ್ದಾಣಕ್ಕೆ"], ["hospital", "ಆಸ್ಪತ್ರೆ", "ಆಸ್ಪತ್ರೆಗೆ"], ["market", "ಮಾರುಕಟ್ಟೆ", "ಮಾರುಕಟ್ಟೆಗೆ"], ["temple", "ದೇವಸ್ಥಾನ", "ದೇವಸ್ಥಾನಕ್ಕೆ"], ["bank", "ಬ್ಯಾಂಕ್", "ಬ್ಯಾಂಕಿಗೆ"], ["post office", "ಅಂಚೆ ಕಚೇರಿ", "ಅಂಚೆ ಕಚೇರಿಗೆ"], ["school", "ಶಾಲೆ", "ಶಾಲೆಗೆ"], ["palace", "ಅರಮನೆ", "ಅರಮನೆಗೆ"], ["airport", "ವಿಮಾನ ನಿಲ್ದಾಣ", "ವಿಮಾನ ನಿಲ್ದಾಣಕ್ಕೆ"]];
const PLACES = [["Mysuru", "ಮೈಸೂರಿಗೆ"], ["Bengaluru", "ಬೆಂಗಳೂರಿಗೆ"], ["Mangaluru", "ಮಂಗಳೂರಿಗೆ"], ["Hubballi", "ಹುಬ್ಬಳ್ಳಿಗೆ"], ["Hassan", "ಹಾಸನಕ್ಕೆ"], ["Dharwad", "ಧಾರವಾಡಕ್ಕೆ"], ["Belagavi", "ಬೆಳಗಾವಿಗೆ"], ["Shivamogga", "ಶಿವಮೊಗ್ಗಕ್ಕೆ"]];
const PRICES = [["ticket", "ಟಿಕೆಟ್"], ["coffee", "ಕಾಫಿ"], ["tea", "ಚಹಾ"], ["dosa", "ದೋಸೆ"], ["idli", "ಇಡ್ಲಿ"], ["juice", "ಜ್ಯೂಸ್"]];
const FIXED = [["Thank you very much for your help.", "ನಿಮ್ಮ ಸಹಾಯಕ್ಕೆ ತುಂಬಾ ಧನ್ಯವಾದಗಳು."], ["Good morning.", "ಶುಭೋದಯ."], ["What is your name?", "ನಿಮ್ಮ ಹೆಸರೇನು?"], ["I do not understand.", "ನನಗೆ ಅರ್ಥವಾಗುತ್ತಿಲ್ಲ."], ["Please speak slowly.", "ದಯವಿಟ್ಟು ನಿಧಾನವಾಗಿ ಮಾತನಾಡಿ."], ["Welcome to Karnataka.", "ಕರ್ನಾಟಕಕ್ಕೆ ಸ್ವಾಗತ."]];
const CORPUS = [];
ITEMS.forEach(([e, k]) => CORPUS.push({ en: `Where is the ${e}?`, kn: `${k} ಎಲ್ಲಿದೆ?`, tag: "where" }));
ITEMS.forEach(([e, k]) => CORPUS.push({ en: `Is there a ${e} near here?`, kn: `ಇಲ್ಲಿ ಹತ್ತಿರ ${k} ಇದೆಯೇ?`, tag: "near" }));
ITEMS.forEach(([e, k, kd]) => CORPUS.push({ en: `Please take me to the ${e}.`, kn: `ದಯವಿಟ್ಟು ನನ್ನನ್ನು ${kd} ಕರೆದುಕೊಂಡು ಹೋಗಿ.`, tag: "take-me" }));
PLACES.forEach(([e, k]) => { CORPUS.push({ en: `I will go to ${e} tomorrow.`, kn: `ನಾನು ನಾಳೆ ${k} ಹೋಗುತ್ತೇನೆ.`, tag: "go-tomorrow" }); CORPUS.push({ en: `I am going to ${e} today.`, kn: `ನಾನು ಇಂದು ${k} ಹೋಗುತ್ತಿದ್ದೇನೆ.`, tag: "go-today" }); });
PRICES.forEach(([e, k]) => CORPUS.push({ en: `How much is the ${e}?`, kn: `${k} ಬೆಲೆ ಎಷ್ಟು?`, tag: "price" }));
PRICES.forEach(([e, k]) => CORPUS.push({ en: `Give me one ${e}.`, kn: `ನನಗೆ ಒಂದು ${k} ಕೊಡಿ.`, tag: "give" }));
PRICES.forEach(([e, k]) => CORPUS.push({ en: `I would like a ${e}.`, kn: `ನನಗೆ ಒಂದು ${k} ಬೇಕು.`, tag: "want" }));
FIXED.forEach(([e, k]) => CORPUS.push({ en: e, kn: k, tag: "fixed" }));
