/* Complete, local periodic-table artifact renderer (118 elements). */
(function () {
  "use strict";

  const Mio = (window.Mio = window.Mio || {});
  const SYMBOLS = "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og".split(" ");
  const NAMES = [
    "Hydrogen", "Helium",
    "Lithium", "Beryllium", "Boron", "Carbon", "Nitrogen", "Oxygen", "Fluorine", "Neon",
    "Sodium", "Magnesium", "Aluminium", "Silicon", "Phosphorus", "Sulfur", "Chlorine", "Argon",
    "Potassium", "Calcium", "Scandium", "Titanium", "Vanadium", "Chromium", "Manganese", "Iron", "Cobalt", "Nickel", "Copper", "Zinc", "Gallium", "Germanium", "Arsenic", "Selenium", "Bromine", "Krypton",
    "Rubidium", "Strontium", "Yttrium", "Zirconium", "Niobium", "Molybdenum", "Technetium", "Ruthenium", "Rhodium", "Palladium", "Silver", "Cadmium", "Indium", "Tin", "Antimony", "Tellurium", "Iodine", "Xenon",
    "Cesium", "Barium", "Lanthanum", "Cerium", "Praseodymium", "Neodymium", "Promethium", "Samarium",
    "Europium", "Gadolinium", "Terbium", "Dysprosium", "Holmium", "Erbium", "Thulium", "Ytterbium", "Lutetium",
    "Hafnium", "Tantalum", "Tungsten", "Rhenium", "Osmium", "Iridium", "Platinum", "Gold", "Mercury",
    "Thallium", "Lead", "Bismuth", "Polonium", "Astatine", "Radon",
    "Francium", "Radium", "Actinium", "Thorium", "Protactinium", "Uranium", "Neptunium", "Plutonium",
    "Americium", "Curium", "Berkelium", "Californium", "Einsteinium", "Fermium", "Mendelevium", "Nobelium", "Lawrencium",
    "Rutherfordium", "Dubnium", "Seaborgium", "Bohrium", "Hassium", "Meitnerium", "Darmstadtium",
    "Roentgenium", "Copernicium", "Nihonium", "Flerovium", "Moscovium", "Livermorium", "Tennessine", "Oganesson",
  ];
  const positions = new Map();

  function place(sequence, row, columns) {
    sequence.split(" ").forEach((symbol, index) => {
      positions.set(symbol, { row, column: columns ? columns[index] : index + 1 });
    });
  }

  place("H He", 1, [1, 18]);
  place("Li Be B C N O F Ne", 2, [1, 2, 13, 14, 15, 16, 17, 18]);
  place("Na Mg Al Si P S Cl Ar", 3, [1, 2, 13, 14, 15, 16, 17, 18]);
  place("K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr", 4);
  place("Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe", 5);
  place("Cs Ba La Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn", 6);
  place("Fr Ra Ac Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og", 7);
  place("Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu", 9, Array.from({ length: 14 }, (_, index) => index + 4));
  place("Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr", 10, Array.from({ length: 14 }, (_, index) => index + 4));

  function category(number, symbol, column) {
    if (number >= 57 && number <= 71) return "lanthanide";
    if (number >= 89 && number <= 103) return "actinide";
    if (column === 18) return "noble";
    if (column === 17) return "halogen";
    if (column === 1 && symbol !== "H") return "alkali";
    if (column === 2) return "alkaline";
    if (["B", "Si", "Ge", "As", "Sb", "Te"].includes(symbol)) return "metalloid";
    if (["H", "C", "N", "O", "P", "S", "Se"].includes(symbol)) return "nonmetal";
    if (column >= 3 && column <= 12) return "transition";
    return "post";
  }

  if (SYMBOLS.length !== 118 || NAMES.length !== 118) {
    throw new Error("Periodic table data must contain 118 elements.");
  }
  const ELEMENTS = Object.freeze(SYMBOLS.map((symbol, index) => {
    const position = positions.get(symbol);
    if (!position) throw new Error("Missing periodic-table position for " + symbol);
    const number = index + 1;
    return Object.freeze({
      symbol,
      number,
      name: NAMES[index],
      row: position.row,
      column: position.column,
      category: category(number, symbol, position.column),
    });
  }));

  function cardsHTML() {
    return ELEMENTS.map((element) => [
      "<button type='button' class='element ", element.category,
      "' style='grid-column:", element.column, ";grid-row:", element.row,
      "' data-number='", element.number, "' data-key='",
      (element.symbol + " " + element.name).toLowerCase(),
      "' aria-label='", element.name, ", atomic number ", element.number, "'>",
      "<span class='number'>", element.number, "</span>",
      "<strong>", element.symbol, "</strong>",
      "<span class='name'>", element.name, "</span>",
      "</button>",
    ].join("")).join("");
  }

  function styles() {
    return [
      ":root{color-scheme:dark}*{box-sizing:border-box}",
      "html,body{margin:0;min-height:100%;background:#0b0d12;color:#edf0f6;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}",
      "body{padding:18px}.toolbar{max-width:1180px;margin:0 auto 12px;display:flex;align-items:center;gap:10px;flex-wrap:wrap}",
      "h1{margin:0;font-size:18px;letter-spacing:-.02em}.meta{color:#8e97a8;font:11px ui-monospace,monospace;flex:1}",
      "input{width:min(280px,100%);border:1px solid #303747;border-radius:8px;background:#151922;color:#edf0f6;padding:8px 10px;outline:none}",
      "input:focus{border-color:#6d8cff;box-shadow:0 0 0 2px #6d8cff33}.scroll{overflow:auto;padding:4px 2px 14px}",
      ".table{min-width:1040px;max-width:1180px;margin:0 auto;display:grid;grid-template-columns:repeat(18,minmax(48px,1fr));grid-template-rows:repeat(10,64px);gap:4px}",
      ".element{min-width:0;border:1px solid #303747;border-radius:7px;background:#181d27;color:#edf0f6;padding:4px;display:flex;flex-direction:column;align-items:center;justify-content:center;cursor:pointer;transition:transform 120ms,border-color 120ms,opacity 120ms}",
      ".element:hover,.element:focus-visible{z-index:2;transform:translateY(-2px) scale(1.05);border-color:#fff9;outline:none}.element[hidden]{display:none}",
      ".number{align-self:flex-start;color:#9aa3b4;font:8px ui-monospace,monospace}.element strong{font-size:16px;line-height:1.1}.name{max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#aab2c1;font-size:8px}",
      ".alkali{background:#3b241d}.alkaline{background:#3a321b}.transition{background:#1c3040}.post{background:#26303b}.metalloid{background:#243a32}.nonmetal{background:#24354a}.halogen{background:#392a46}.noble{background:#2b294c}.lanthanide{background:#3c2838}.actinide{background:#43282e}",
      ".detail{max-width:1180px;min-height:54px;margin:4px auto 0;padding:12px 14px;border:1px solid #2c3341;border-radius:10px;background:#131720;display:flex;gap:14px;align-items:center}",
      ".detail strong{font-size:22px}.detail span{color:#aab2c1;font-size:12px}.empty{color:#ffbdc7}",
      "@media(max-width:600px){body{padding:12px}.toolbar{align-items:stretch}.toolbar input{width:100%;order:2}.table{grid-template-rows:repeat(10,58px)}.detail{position:sticky;left:0}}",
    ].join("");
  }

  function runtime() {
    const data = JSON.stringify(ELEMENTS).replace(/</g, "\\u003c");
    return [
      "<script>",
      "'use strict';",
      "const elements=", data, ";",
      "const cards=Array.from(document.querySelectorAll('.element'));",
      "const query=document.getElementById('query');",
      "const symbol=document.getElementById('detail-symbol');",
      "const description=document.getElementById('detail-description');",
      "const categoryNames={alkali:'alkali metal',alkaline:'alkaline-earth metal',transition:'transition metal',post:'post-transition element',metalloid:'metalloid',nonmetal:'reactive nonmetal',halogen:'halogen',noble:'noble gas',lanthanide:'lanthanide',actinide:'actinide'};",
      "function show(card){const item=elements[Number(card.dataset.number)-1];if(!item)return;symbol.textContent=item.symbol;description.textContent=item.name+' · atomic number '+item.number+' · '+categoryNames[item.category]+' · '+(item.row>8?'f-block series':'period '+item.row+', group '+item.column);cards.forEach(value=>value.setAttribute('aria-pressed',String(value===card)));}",
      "cards.forEach(card=>card.addEventListener('click',()=>show(card)));",
      "query.addEventListener('input',()=>{const value=query.value.trim().toLowerCase();let visible=0;cards.forEach(card=>{card.hidden=Boolean(value)&&!card.dataset.key.includes(value)&&card.dataset.number!==value;if(!card.hidden)visible++;});if(!visible){symbol.textContent='—';description.textContent='No element matches “'+query.value+'”.';description.className='empty';}else{description.className='';}});",
      "if(cards.length)show(cards[0]);",
      "</", "script>",
    ].join("");
  }

  function template() {
    return [
      "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
      "<meta name='viewport' content='width=device-width,initial-scale=1'>",
      "<title>Periodic table</title><style>", styles(), "</style></head><body>",
      "<header class='toolbar'><h1>Periodic table</h1><span class='meta'>118 elements · complete local dataset</span>",
      "<input id='query' type='search' autocomplete='off' placeholder='Search name, symbol, or number' aria-label='Search elements'></header>",
      "<main class='scroll'><section class='table' aria-label='Periodic table of elements'>",
      cardsHTML(),
      "</section></main>",
      "<section class='detail' aria-live='polite'><strong id='detail-symbol'>H</strong><span id='detail-description'>Hydrogen</span></section>",
      runtime(),
      "</body></html>",
    ].join("");
  }

  Mio.artifactPeriodic = Object.freeze({
    count: () => SYMBOLS.length,
    template,
  });
})();
