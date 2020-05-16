// TESTER = document.getElementById('tester');
// 	Plotly.newPlot( TESTER, [{
// 	x: [[1, 2, 3, 4, 5,6],[1,2,3,4,5,6]],
// 	y: [[1, 2, 4, 8, 16,15],[3,2,1,5,6,7]] }], {
//     margin: { t: 0 } } );
    
var mydata = JSON.parse(bairros);

dates = [1,  2,  3,  4,  5,  6,  7,  8,  9,  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49];

console.log(mydata);
  var lines = [];

  for (i = 0; i < 2; i++) {
    console.log(mydata[i].Nome);
    lines.push({x: dates, y: mydata[i].Casos, name: mydata[i].Nome});
  };
    // var trace1 = {
    //     x: a,
    //     y: [10, 15, 13, 17],
    //     type: 'scatter'
    //   };
      
    //   var trace2 = {
    //     x: [1, 2, 3, 4],
    //     y: [16, 5, 11, 9],
    //     type: 'scatter',
    //     name: 'Alow'
    //   };
      
    //   var data = [trace1, trace2];
      
     Plotly.newPlot('tester', lines);