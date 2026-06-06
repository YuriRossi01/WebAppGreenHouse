import { getGradientCold, getGradientHot,refreshHeatLayers ,updateTemperatures,parsefloatGlobalData,updateSliderMax,playSlider } from 'constFunction';

// variabili glocali
const mymap = L.map("map");
mymap.setView([45.402866, 10.998162], 19);
var tileLayer = L.tileLayer(
  "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
  //"https://tiles.stadiamaps.com/tiles/osm_bright/{z}/{x}/{y}{r}.png",
  {
    attribution:
      '&copy; <a href="https://www.stadiamaps.com/" target="_blank">Stadia Maps</a> &copy; <a href="https://openmaptiles.org/" target="_blank">OpenMapTiles</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    //ext: 'png'
    maxZoom: 19,
    minZoom: 16,
    maxNativeZoom: 19
  }
);
tileLayer.addTo(mymap);

var legend = L.control({ position:"bottomleft"});

legend.onAdd = function(mymap) {
  var div = L.DomUtil.create("div","legend");
  div.innerHTML += "<h4>Legenda Termica</h4>";

    div.innerHTML += '<i style="background: #ff0000"></i><span>&gt;= 30°C</span><br>';
    div.innerHTML += '<i style="background: #ff7f00"></i><span>25°C a 30°C</span><br>';
    div.innerHTML += '<i style="background: #ffff00"></i><span>20°C a 25°C</span><br>';
    div.innerHTML += '<i style="background: #00ff00"></i><span>15°C a 20°C</span><br>';
    div.innerHTML += '<i style="background: #00ffff"></i><span>10°C a 15°C</span><br>';
    div.innerHTML += '<i style="background: #0000ff"></i><span>0°C a 10°C</span><br>';
    div.innerHTML += '<i style="background: #000080"></i><span>&lt;= 0°C</span><br>';

  return div;
};

legend.addTo(mymap);
var globalData = [];
// Coordinates of the sensors and their detected temperature
var temperature = [
  [45.402866, 10.998162, -10], // lat, lng, (temperature) intensity
  [45.40265, 10.99828, -11],
  [45.40269, 10.99845, -12],
  [45.4029, 10.99831, -13],
  [45.40286, 10.99841, -14],
  [45.40276, 10.99832, -15],
  [45.40279, 10.99823, -16],
  [45.40279, 10.99823, -17]
];
let text = '[]';
let temperatureReq = JSON.parse(text);
var coldGradient = [1, 1, 1, 1, 1, 1, 1, 1, 1];


// mettere altro file
/*
function getGradientCold(tempTemp) {
  if (tempTemp <= 0) return 1.0;
  else if (tempTemp > 0 && tempTemp <= 5) return 1 - tempTemp / 30;
  else if (tempTemp > 5 && tempTemp <= 10) return 1 - tempTemp / 30;
  else if (tempTemp > 10 && tempTemp <= 15) return 1 - tempTemp / 30;
}
  */
// mettere altro file
/*
function getGradientHot(tempTemp) {
  if (tempTemp >= 30) return 1.0;
  else if (tempTemp >= 15 && tempTemp <= 20) return tempTemp / 30.0;
  else if (tempTemp > 20 && tempTemp <= 25) return tempTemp / 30.0;
  else if (tempTemp > 25 && tempTemp <= 30) return tempTemp / 30.0;
}
*/


// Initialize heat layers (cold)
var heatColdPoints = L.heatLayer(temperature, {
  radius: 20,
  minOpacity: 0.7,
  gradient: {
    0.0: "red",
    0.14: "orange",
    0.28: "#FFBF00",
    0.42: "yellow",
    0.56: "lime",
    0.7: "aqua",
    0.84: "blue",
    1.0: "black"
  }
});
heatColdPoints.addTo(mymap); // add heatColdlayer in to the map
// Initialize heat layers (Hot)
var heatHotPoints = L.heatLayer(temperature, {
  radius: 20,
  minOpacity: 0.7,
  gradient: {
    0.0: "black",
    0.14: "blue",
    0.28: "aqua",
    0.42: "lime",
    0.56: "yellow",
    0.7: "#FFBF00",
    0.84: "orange",
    1.0: "red"
  }
});
heatHotPoints.addTo(mymap);

// section button reset:
  // costant
const resetButton = document.getElementById("reset"); // id reset map button
function resetBtn(){
    // Reset della mappa
    mymap.setView([45.402866, 10.998162], 19);

    // Reset dei campi input
    document.getElementById("chooseData").value = "";
    document.getElementById("time").value = "";
    document.getElementById("model").selectedIndex = 0; // Torna alla prima opzione
}
resetButton.addEventListener("click", resetBtn);

function limitationDate() {
  let dateInput = document.getElementById("chooseData");
  let today = new Date(); // Data corrente
  let tenYearsAgo = new Date(); // Calcolo 10 anni indietro
  tenYearsAgo.setFullYear(today.getFullYear() - 10);

  // Converti le date nel formato YYYY-MM-DD
  const minDate = tenYearsAgo.toISOString().split("T")[0];

  // Imposta gli attributi di limite
  dateInput.setAttribute("min", minDate);
}

// secxtion choose date:

const changeMap = document.getElementById("updateMapBtn")
function chooseData() {
  updateMapBtn.disabled = true; // disabilitato
  let loadingOverlay = document.getElementById("loading-overlay");
  loadingOverlay.style.display = "block"; // Mostra la GIF di caricamento
  let arrAlert = [0, 0, 0, 0, 0, 0, 0];
  let dateInput = document.getElementById("chooseData").value;  // Prende il valore in formato YYYY-MM-DD
  let timeInput = document.getElementById("time").value;  // Prende il valore in formato HH:MM
  let chooseBox = document.getElementById("model").value;  // Prende il valore selezionato

  // Verifica se la data e l'ora sono vuote
  if (timeInput === "" || dateInput === "") {
    arrAlert[0] = 1;
  }
  let errorMessage = generateAlert(arrAlert);
  if (errorMessage) {
    updateMapBtn.disabled = true; // abilitato
    loadingOverlay.style.display = "none"; // Nasconde la GIF
    alert(errorMessage);

  }else {
    // Crea l'URL della richiesta
    const xhr = new XMLHttpRequest();
    let url = "http://localhost:8000/reqTemp/?day=" + encodeURIComponent(dateInput) + "&model=" + chooseBox + "&time=" + encodeURIComponent(timeInput)
    xhr.open('GET', url);
    xhr.responseType = "json";
    xhr.onload = function() {
      if (xhr.readyState == 4 && xhr.status == 200) {
        updateMapBtn.disabled = true; // disabilitato
        loadingOverlay.style.display = "none"; // Nasconde la GIF
        let i=0;
        // Aggiorna i dati dei sensori se la richiesta ha successo0
        let data = xhr.response;

        globalData = parsefloatGlobalData(data,globalData);

        updateTemperatures(mymap,globalData,null,temperature, heatColdPoints, heatHotPoints);

      } else {
        arrAlert[1] = 1;
        generateAlert(arrAlert);
        //updateMapBtn.disabled = false;
        loadingOverlay.style.display = "none"; // Nasconde la GIF
      }
    };
    xhr.onerror = function () { // Errore nella richiesta.
      //updateMapBtn.disabled = false;
      loadingOverlay.style.display = "none"; // Nasconde la GIF in caso di errore
      arrAlert[2] = 1;
      generateAlert(arrAlert);
    };
    // Invia la richiesta
    xhr.send();
    updateMapBtn.disabled = false;
  }
}
changeMap.addEventListener("click",chooseData)

function generateAlert(arrAlert) {
  let errorMessages = [];
  let errorMessagesList = [
    'Data e / o tempo non selezionate.',
    'Errore recupero Dati.',
    'Errore nella richiesta.',
  ];
  for (let i = 0; i < arrAlert.length; i++) {
    if (arrAlert[i] === 1) {
      errorMessages.push(errorMessagesList[i]);
    }
  }
  return errorMessages.length > 0 ? errorMessages.join('; ') + '.' : '';
}
// mettere altro file

function getArrTempToStr(data){
  let dataFloat = JSON.parse(data);

  //let str = data.substring(2,data.length-2);
  globalData = [];

  for( let i = 0; i < dataFloat.length; i++){
    if(dataFloat[i]){
      let convert = parseFloat(dataFloat[i]);
      if(!isNaN(convert)){
        arr.push(convert);
      }
    }
  }

  for (value of dataFloat.split(".")){
    arr.push(parseFloat(value).toFixed(3));
  }
  return arr;
}

// ----------- Prints section --------------\\

// Function to update temperature labels
function initializeTable() {
  for (var i = 0; i < temperature.length; i++) {
    var valueTemperature = temperature[i][2];
    document.getElementById("sens" + (i + 1)).innerText = valueTemperature.toFixed(2) + "°C";
    //document.getElementById("sens" + (i + 1)).innerText = temperature[i][2]+"°C";
  }
}
// Initial setup
initializeTable(); // Display table initially
limitationDate();
