let isPlaying = false;
let interval;

function getGradientCold(tempTemp) {
  if (tempTemp <= 0) return 1.0;
  else if (tempTemp > 0 && tempTemp <= 5) return 1 - tempTemp / 30;
  else if (tempTemp > 5 && tempTemp <= 10) return 1 - tempTemp / 30;
  else if (tempTemp > 10 && tempTemp <= 15) return 1 - tempTemp / 30;
}

function getGradientHot(tempTemp) {
  if (tempTemp >= 30) return 1.0;
  else if (tempTemp >= 15 && tempTemp <= 20) return tempTemp / 30.0;
  else if (tempTemp > 20 && tempTemp <= 25) return tempTemp / 30.0;
  else if (tempTemp > 25 && tempTemp <= 30) return tempTemp / 30.0;
}

// Function to refresh heat layers
function refreshHeatLayers(mymap,arrHot,arrCold,heatColdPoints,heatHotPoints) {
  Object.values(mymap._layers).forEach(layer => {
    if (!layer._url) {
        mymap.removeLayer(layer);
    }
  });

  for (let i = 0; i < arrHot.length; i++)
    arrHot[i][2] = getGradientHot(arrHot[i][2]);

  for (let j = 0; j < arrCold.length; j++)
    arrCold[j][2] = getGradientCold(arrCold[j][2]);

  heatHotPoints = L.heatLayer(arrHot, {
    radius: 20,
    gradient: {

      0.5: "yellow",
      0.66: "#FFBF00",
      0.82: "orange",
      0.98: "red"
    }
  }).addTo(mymap);

  heatColdPoints = L.heatLayer(arrCold, {
    radius: 20,
    minOpacity: 0.7,
    gradient: {

      0.5: "lime",
      0.66: "aqua",
      0.98: "blue",
      0.98: "#0000CD"

    }
  }).addTo(mymap);


  let count = 0;
  mymap.eachLayer(function(layer) {
    count++;
  });

}

function updateTemperatures(mymap, globalData, changeValue, temperature, heatColdPoints, heatHotPoints) {
  let arrCold = [];
  let arrHot = [];
  let sens = 0;

  // Aggiorna le temperature
  if (changeValue != null) {
    if (globalData && globalData.length > 0 && changeValue >= 0 && changeValue < globalData[0].length) {
      for (let i = 0; i < globalData.length; i++) {
        temperature[i][2] = globalData[i][changeValue]; // Aggiorna le temperature
      }
    }
  } else {
    for (let i = 0; i < globalData.length; i++) {
      temperature[i][2] = globalData[i][0]; // Aggiorna le temperature
    }
  }

  // Aggiorna la tabella e smista le temperature in arrCold e arrHot
  temperature.forEach(function (rigaT) {
    sens++;
    let temptemp = rigaT[2];
    document.getElementById("sens" + sens).innerText = temptemp.toFixed(2) + "°C"; // Aggiorna la cella della tabella

    // Crea un array con le coordinate e la temperatura
    let rigaTemp = [rigaT[0], rigaT[1], temptemp]; // lat, lng, temperature
    if (temptemp < 15) {
      arrCold.push(rigaTemp); // Aggiungi a arrCold se la temperatura è sotto i 15°C
    } else {
      arrHot.push(rigaTemp); // Aggiungi a arrHot se la temperatura è 15°C o superiore
    }
  });

  // Aggiorna i layer di calore sulla mappa
  refreshHeatLayers(mymap, arrHot, arrCold, heatColdPoints, heatHotPoints);
}
function parsefloatGlobalData(data,globalData) {
  let dataFloat = JSON.parse(data); // Converti la stringa JSON in un array
  globalData = Array.from({ length: 8 }, () => []); // Inizializza un array multidimensionale per 8 sensori

  for (let i = 0; i < dataFloat.length; i++) {
      if (dataFloat[i] && Array.isArray(dataFloat[i])) {
          for (let j = 0; j < dataFloat[i].length; j++) {
              if (dataFloat[i][j] && Array.isArray(dataFloat[i][j]) && dataFloat[i][j][0] !== undefined) {
                  // Estrai la temperatura dal formato [[[temperature]]]
                  let convert = parseFloat(dataFloat[i][j][0]); // Prendi il primo elemento dell'array interno
                  if (!isNaN(convert)) {
                      globalData[i].push(convert); // Aggiungi la temperatura all'array del sensore corrispondente
                  }
              }
          }
      }
  }

  // Aggiorna il massimo dello slider
  let slider = document.querySelector(".slider");
  if (slider)
    updateSliderMax(globalData[0].length - 1);

  return globalData;
}

function updateSliderMax(len) {
  // Seleziona l'elemento slider usando la classe
  let slider = document.querySelector(".slider");
  if (!slider) {
    console.error("Elemento slider con classe 'slider' non trovato!");
    return;
  }
  slider.max = len; // Aggiorna il massimo
  slider.setAttribute("data-max", len); // Aggiorna data-max
  slider.value = 0; // Ripristina il valore dello slider
  document.getElementById("value").innerText = slider.value; // Mostra il valore iniziale
}

function playSlider(playButton, mymap, globalData, temperature, heatColdPoints, heatHotPoints) {
   console.log("Stato GlobalData:", globalData);
   let speedElement = document.getElementById("sliderSpeed");
   let slider = document.getElementById("myRange");
   let valueSlider = document.getElementById("value");

   if (isPlaying) {
     clearInterval(interval);
     playButton.textContent = "Play";
   } else {
     if (parseInt(slider.value) >= parseInt(slider.max)) {
         slider.value = 0;
         valueSlider.textContent = 0;
     }
     let currentSpeed = parseFloat(speedElement.value) || 1;
     interval = setInterval(function () {
       intervalTemp(slider, valueSlider, playButton, mymap, globalData, temperature, heatColdPoints, heatHotPoints);
     }, 1000 / currentSpeed);

     playButton.textContent = "Pause";
   }
   isPlaying = !isPlaying;
 }

function intervalTemp(slider, valueSlider, playButton, mymap, globalData, temperature, heatColdPoints, heatHotPoints) {
  if (parseInt(slider.value) < parseInt(slider.max)) {
    slider.value = parseInt(slider.value) + 1; // Incrementa il valore dello slider
    valueSlider.textContent = slider.value; // Aggiorna il display

    updateTemperatures(mymap, globalData, parseInt(slider.value), temperature, heatColdPoints, heatHotPoints);
  } else {
    clearInterval(interval); // Ferma il ciclo
    playButton.textContent = "Play"; // Cambia il testo del pulsante
    isPlaying = false; // Aggiorna il flag
  }
}


export { getGradientCold,getGradientHot,refreshHeatLayers,updateTemperatures,parsefloatGlobalData,updateSliderMax,playSlider,intervalTemp};