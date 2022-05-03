var modelData;

var listOfRows = ["0", "1", "2", "accr", "mavg", "wavg"]


var AJAXtojson = $.ajax({
    type: 'GET',
    url: '/getmodeldata',
    async: true,
    success: function(response) { 
        modelData = JSON.parse(JSON.stringify(response));
        console.log(modelData);

        var p = document.getElementById("card-text");
        p.innerHTML = "Accuracy for this model is:  " + modelData["accuracy"];

        for (var i = 0; i < modelData["data"].length; i++){

            var row = document.getElementById(listOfRows[i]);
            for (var j = 0; j < modelData["data"][i].length; j++){

                var td = document.createElement("td");
                td.innerHTML = modelData["data"][i][j];
                row.appendChild(td);
            }
        }
    },
    contentType: "application/json",
});


