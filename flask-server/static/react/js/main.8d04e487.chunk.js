(this["webpackJsonpstock-game"]=this["webpackJsonpstock-game"]||[]).push([[0],{54:function(e,t,a){e.exports=a(72)},59:function(e,t,a){},60:function(e,t,a){},61:function(e,t,a){},68:function(e,t,a){},69:function(e,t,a){},70:function(e,t,a){},72:function(e,t,a){"use strict";a.r(t);var r=a(0),n=a.n(r),i=a(42),s=a.n(i),l=(a(59),a(21)),c=a(22),o=a(28),d=a(27),m=(a(60),a(1)),u=a(17);a(61);var h=function(){return n.a.createElement("svg",{id:"robot-alt",width:"512",height:"512",viewBox:"0 0 512 512",fill:"none",xmlns:"http://www.w3.org/2000/svg"},n.a.createElement("rect",{width:"512",height:"512"}),n.a.createElement("rect",{x:"10.5",y:"232.618",width:"53.3529",height:"184.647",rx:"20.5",stroke:"#F0F4F6","stroke-width":"5"}),n.a.createElement("rect",{x:"448.147",y:"232.618",width:"53.3529",height:"184.647",rx:"20.5",stroke:"#F0F4F6","stroke-width":"5"}),n.a.createElement("rect",{x:"185.559",y:"169.265",width:"53.3529",height:"140.882",rx:"20.5",transform:"rotate(-90 185.559 169.265)",stroke:"#F0F4F6","stroke-width":"5"}),n.a.createElement("rect",{x:"39.6765",y:"145.088",width:"432.647",height:"359.706",rx:"47.5",fill:"#1e2124",stroke:"#F0F4F6","stroke-width":"5"}),n.a.createElement("circle",{cx:"350.824",cy:"273.882",r:"48.5588",stroke:"#F0F4F6","stroke-width":"5"}),n.a.createElement("rect",{x:"163.676",y:"417.265",width:"53.3529",height:"184.647",rx:"20.5",transform:"rotate(-90 163.676 417.265)",stroke:"#F0F4F6","stroke-width":"5"}),n.a.createElement("circle",{cx:"153.882",cy:"273.882",r:"63.1471",stroke:"#F0F4F6","stroke-width":"5"}),n.a.createElement("rect",{x:"247.5",y:"74.5",width:"17",height:"41",stroke:"#F0F4F6","stroke-width":"5"}),n.a.createElement("circle",{cx:"256",cy:"40.4706",r:"33.9706",stroke:"#F0F4F6","stroke-width":"5"}))};var p=function(){return n.a.createElement("div",{id:"homeContainer",className:"routeContainer"},n.a.createElement("div",{id:"homeTitleContainer"},n.a.createElement(h,null),n.a.createElement("h1",{id:"homeTitle"},"StockBOT")),n.a.createElement("br",null),n.a.createElement("p",null,"Welcome to StockBOT! You will be pitted against a machine-learning algorithm to see who can get the biggest return on investment. Here's how the game is played:"),n.a.createElement("ol",{id:"gameRules"},n.a.createElement("li",null,'A random stock and year will be chosen when you press "Start."'),n.a.createElement("li",null,'The graph will start trending, and you can press "Buy" to purchase a stock on the current date.'),n.a.createElement("li",null,'Once a stock has been purchased you can sell the stock on the current date by pressing "Sell," or you can hold the stock by not pressing anything and waiting for time to run out.'),n.a.createElement("li",null,"Once time has ended, you will be ranked alongside the machine-learning algorithm and the market.")),n.a.createElement("br",null),n.a.createElement("p",{id:"home-accent-paragraph"},"May the best trader win!"),n.a.createElement("br",null),n.a.createElement(u.b,{to:"/game"},n.a.createElement("button",{id:"playButton"},"Play")))},y=a(19),g=a.n(y),f=a(44),v=a(9);a(68),a(69);var b=function(){return n.a.createElement("div",{id:"legendContainer"},n.a.createElement("h1",{id:"legendTitle"},"Legend"),n.a.createElement("div",{className:"legendRow"},n.a.createElement("svg",{className:"svgCircle",width:"14",height:"14"},n.a.createElement("circle",{id:"playerBuyLegend",cx:"7",cy:"7",r:"7"})),n.a.createElement("div",{className:"legendText"},"Player Buy")),n.a.createElement("div",{className:"legendRow"},n.a.createElement("svg",{className:"svgCircle",width:"14",height:"14"},n.a.createElement("circle",{id:"playerSellLegend",cx:"7",cy:"7",r:"7"})),n.a.createElement("div",{className:"legendText"},"Player Sell")),n.a.createElement("div",{className:"legendRow"},n.a.createElement("svg",{className:"svgCircle",width:"16",height:"16"},n.a.createElement("circle",{id:"aiBuyLegend",cx:"8",cy:"8",r:"7"})),n.a.createElement("div",{className:"legendText"},"AI Buy")),n.a.createElement("div",{className:"legendRow"},n.a.createElement("svg",{className:"svgCircle",width:"16",height:"16"},n.a.createElement("circle",{id:"aiSellLegend",cx:"8",cy:"8",r:"7"})),n.a.createElement("div",{className:"legendText"},"AI Sell")))};a(70);var E=function(e){var t=new Date(e.time);return"number"===typeof e.time&&(t=new Intl.DateTimeFormat("en-US").format(t)),n.a.createElement("div",{id:"scoreboardContainer"},n.a.createElement("h1",{id:"scoreboardTitle"},"Stats"),n.a.createElement("div",{className:"scoreboard-row"},n.a.createElement("span",{id:"current-date",className:"stats"},"Current Date"),n.a.createElement("div",{className:"statsText time"},t.toString())),n.a.createElement("div",{className:"scoreboard-row"},n.a.createElement("span",{className:"stats"},"Current Price"),n.a.createElement("div",{className:"statsGroup"},n.a.createElement("div",{className:"statsText price"},e.price?"".concat(new Intl.NumberFormat("en-US",{style:"currency",currency:"USD"}).format(e.price)):""))),n.a.createElement("div",{className:"scoreboard-row"},n.a.createElement("span",{className:"stats"},"Player Buy"),n.a.createElement("div",{className:"statsText price buy"},e.playerBuys.price?"".concat(new Intl.NumberFormat("en-US",{style:"currency",currency:"USD"}).format(e.playerBuys.price)):"")),n.a.createElement("div",{className:"scoreboard-row"},n.a.createElement("span",{className:"stats"},"Player Sell"),n.a.createElement("div",{className:"statsText price sell"},e.playerSells.price?"".concat(new Intl.NumberFormat("en-US",{style:"currency",currency:"USD"}).format(e.playerSells.price)):"")),n.a.createElement("div",{className:"scoreboard-row"},n.a.createElement("span",{className:"stats"},"AI Buy"),n.a.createElement("div",{className:"statsText price buy"},e.aiBuys.price&&e.time>=e.aiBuys.time?"".concat(new Intl.NumberFormat("en-US",{style:"currency",currency:"USD"}).format(e.aiBuys.price)):"")),n.a.createElement("div",{className:"scoreboard-row"},n.a.createElement("span",{className:"stats"},"AI Sell"),n.a.createElement("div",{className:"statsText price sell"},e.aiSells.price&&e.time>=e.aiSells.time?"".concat(new Intl.NumberFormat("en-US",{style:"currency",currency:"USD"}).format(e.aiSells.price)):"")))},w=a(82),k=a(84),S=a(80),x=a(81),B=a(79),N=a(36),C=a(85),L=a(83),F=(a(71),function(e){Object(o.a)(a,e);var t=Object(d.a)(a);function a(e){var r;return Object(l.a)(this,a),(r=t.call(this,e)).state={speed:.5,data:null,dataIndex:0,timeframe:"ALL",dataTime:"",dataPrice:"",previousPrice:"",playerBuys:{time:"",price:""},playerSells:{time:"",price:""},aiBuys:{time:"",price:""},aiSells:{time:"",price:""}},r.createChart=r.createChart.bind(Object(v.a)(r)),r.animateChart=r.animateChart.bind(Object(v.a)(r)),r.handleClick=r.handleClick.bind(Object(v.a)(r)),r.handleSlide=r.handleSlide.bind(Object(v.a)(r)),r.createResults=r.createResults.bind(Object(v.a)(r)),r.handleLowerBoundChange=r.handleLowerBoundChange.bind(Object(v.a)(r)),r.chartRef=n.a.createRef(),r}return Object(c.a)(a,[{key:"componentDidMount",value:function(){var e=this;fetch("/data").then((function(t){t.ok&&t.json().then((function(t){var a={data:JSON.parse(t.data),stock:t.stock},r=a.data.data.map((function(t,r){if(1===t[1]&&!e.state.aiBuys.time)return{time:a.data.index[r],price:t[0]}})).filter((function(e){if(e)return e})),n=a.data.data.map((function(t,r){if(-1===t[1]&&!e.state.aiSells.time)return{time:a.data.index[r],price:t[0]}})).filter((function(e){if(e&&e.time>r[0].time)return e}));e.setState({data:a,dataTime:a.data.index[0],aiBuys:r[0]?r[0]:"",aiSells:n[0]?n[0]:""},e.createChart)}))}))}},{key:"createChart",value:function(){var e=this.chartRef.current,t=this.state.data,a=t.data,r=(t.stock,a.index.map((function(e,t){return[e,a.data[t]]}))),n={top:0,right:0,bottom:60,left:0},i=Object(w.a)().range([n.left,890-n.right-n.left]),s=Object(k.a)().range([n.top,500-n.bottom-n.top]),l=Object(B.a)(e).append("g").attr("id","xAxis").attr("transform","translate(0, "+(500-n.bottom)+")"),c=Object(B.a)(e).append("g").attr("id","yAxis").attr("transform","translate("+n.left+", 0)"),o=C.a().x((function(e){return i(e[0])})).y((function(e){return s(e[1][0])})),d=Object(B.a)(e).append("path").attr("id","path"),m=Object(B.a)(e).selectAll("circle").data(r).enter().append("circle").attr("cx",(function(e){return i(e[0])})).attr("cy",(function(e){return s(e[1][0])}));this.animateChart(e,i,s,l,c,o,d,m,n,890,500)}},{key:"animateChart",value:function(){var e=Object(f.a)(g.a.mark((function e(t,a,r,n,i,s,l,c,o,d,m){var u,h,p,y,f,v,b=this;return g.a.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:u=this.state.data,h=u.data,u.stock,p=h.index.map((function(e,t){return[e,h.data[t]]})),y=function(e){return new Promise((function(t){return setTimeout(t,e)}))},f=g.a.mark((function e(t){var m,u;return g.a.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:m=void 0,e.t0=b.state.timeframe,e.next="ALL"===e.t0?4:"1D"===e.t0?6:"1W"===e.t0?8:"1M"===e.t0?10:"3M"===e.t0?12:14;break;case 4:return m=0,e.abrupt("break",16);case 6:return m=t-1,e.abrupt("break",16);case 8:return m=t-7,e.abrupt("break",16);case 10:return m=t-22,e.abrupt("break",16);case 12:return m=t-66,e.abrupt("break",16);case 14:return m=0,e.abrupt("break",16);case 16:return console.log("lower index = ",m),u=p.slice(m,t+1),b.setState((function(e){return{dataIndex:t,dataTime:u[u.length-1][0],dataPrice:u[u.length-1][1][0],previousPrice:e.dataPrice}})),a.domain([p[m][0],p[t][0]]),r.domain([Object(S.a)(u,(function(e){return e[1][0]}))+5,Object(x.a)(u,(function(e){return e[1][0]}))-5]),n.call(Object(N.a)(a).tickSizeOuter(0)),i.call(Object(N.b)(r).tickSize(d-o.left-o.right).tickFormat(Object(L.a)(".2f"))),l.attr("d",s(p)),c.attr("cx",(function(e){return a(e[0])})).attr("cy",(function(e){return r(e[1][0])})).attr("class",(function(e){return e[0]===b.state.playerBuys.time?"dot playerBuy":e[0]===b.state.playerSells.time&&b.state.playerBuys.time?"dot playerSell":e[0]===b.state.aiBuys.time?"dot aiBuy":e[0]===b.state.aiSells.time&&b.state.aiBuys.time?"dot aiSell":"dot"})).attr("r",7),e.next=27,y(1e3*(1-b.state.speed));case 27:case"end":return e.stop()}}),e)})),v=0;case 5:if(!(v<p.length)){e.next=10;break}return e.delegateYield(f(v),"t0",7);case 7:v++,e.next=5;break;case 10:this.createResults();case 11:case"end":return e.stop()}}),e,this)})));return function(t,a,r,n,i,s,l,c,o,d,m){return e.apply(this,arguments)}}()},{key:"createResults",value:function(){window.outerWidth;var e=(window.outerHeight-250)/2,t={who:"Player",return:0},a={who:"AI",return:0};if(this.state.playerBuys.price){var r=this.state.playerBuys.price,n=this.state.playerSells.price?this.state.playerSells.price:this.state.dataPrice;t.return=(n-r)/r}if(this.state.aiBuys.price){var i=this.state.aiBuys.price,s=this.state.aiSells.price?this.state.aiSells.price:this.state.dataPrice;a.return=(s-i)/i}var l=this.state.data.data.data[0][0],c={who:"Market",return:(this.state.dataPrice-l)/l};Object(B.a)("#chartContainer").append("div").html((function(){var e=[t,a,c].sort((function(e,t){return t.return-e.return})).map((function(e){return e.return=Object(L.a)(".2%")(e.return),e}));return'<div id="resultsTitle">Results</div>\n          <ul id="results-list">\n            <li class="list-item">\n              <span class="player-name">1. '.concat(e[0].who,'</span> \n              <span class="player-percent">').concat(e[0].return,'</span>\n            </li>\n            <li class="list-item">\n              <span class="player-name">2. ').concat(e[1].who,'</span> \n              <span class="player-percent">').concat(e[1].return,'</span>\n            </li>\n            <li class="list-item">\n              <span class="player-name">3. ').concat(e[2].who,'</span> \n              <span class="player-percent">').concat(e[2].return,'</span>\n            </li>\n          </ul>\n          <button onclick="window.location.reload()" id="resultsButton">Play Again</button>')})).attr("id","resultsHTML").style("top",-e+"px").transition().duration(1e3).style("top",e+"px"),Object(B.a)("#root").append("div").attr("id","darken-on-results")}},{key:"handleClick",value:function(e){console.log("e.target button: ",e.target.textContent),"Buy"!==e.target.textContent||this.state.playerBuys.time||(console.log("state set to BUY"),this.setState((function(e){return{playerBuys:{time:e.data.data.index[e.dataIndex],price:e.data.data.data[e.dataIndex][0]}}}),console.log(this.state))),"Sell"===e.target.textContent&&this.state.playerBuys.time&&!this.state.playerSells.time&&(console.log("state set to SELL"),this.setState((function(e){return{playerSells:{time:e.data.data.index[e.dataIndex],price:e.data.data.data[e.dataIndex][0]}}}),console.log(this.state)))}},{key:"handleSlide",value:function(e){console.log("inside handleSlide"),console.log("e.target.value:",e.target.value),this.setState({speed:e.target.value})}},{key:"handleLowerBoundChange",value:function(e){var t=this.state.dataIndex;switch(e.target.textContent){case"ALL":console.log("state set to ALL"),this.setState({timeframe:"ALL"});break;case"1D":t-1>0&&(console.log("state set to 1D"),this.setState({timeframe:"1D"}));break;case"1W":t-5>0&&(console.log("state set to 1W"),this.setState({timeframe:"1W"}));break;case"1M":t-22>0&&(console.log("state set to 1M"),this.setState({timeframe:"1M"}));break;case"3M":t-66>0&&(console.log("state set to 3M"),this.setState({timeframe:"3M"}));break;default:console.log("default chosen set to ALL"),this.setState({timeframe:"ALL"})}}},{key:"render",value:function(){return n.a.createElement("div",{id:"chartContainer"},n.a.createElement("div",{className:"col",id:"col1"},n.a.createElement(E,{time:this.state.dataTime,price:this.state.dataPrice,previousPrice:this.state.previousPrice,playerBuys:this.state.playerBuys,playerSells:this.state.playerSells,aiBuys:this.state.aiBuys,aiSells:this.state.aiSells}),n.a.createElement(b,null)),n.a.createElement("div",{className:"col",id:"col2"},n.a.createElement("div",{id:"chart-title-container"},n.a.createElement("h2",{id:"chartTitle"},"".concat(this.state.data?this.state.data.stock[1]:""," ").concat(this.state.data?"("+this.state.data.stock[0]+")":"")),n.a.createElement("div",{id:"chart-title-price"},"".concat(this.state.dataPrice?new Intl.NumberFormat("en-US",{style:"currency",currency:"USD"}).format(this.state.dataPrice):""))),n.a.createElement("svg",{id:"chart",ref:this.chartRef,width:"890px",height:"500px"}),n.a.createElement("div",{id:"controlsContainer"},n.a.createElement("div",{className:"control-row one"},n.a.createElement("ul",{id:"domain-toggle-list"},n.a.createElement("li",{id:"1D-li",className:"1D"===this.state.timeframe?"active-list":"",onClick:this.handleLowerBoundChange},"1D"),n.a.createElement("li",{id:"1W-li",className:"1W"===this.state.timeframe?"active-list":"",onClick:this.handleLowerBoundChange},"1W"),n.a.createElement("li",{id:"1M-li",className:"1M"===this.state.timeframe?"active-list":"",onClick:this.handleLowerBoundChange},"1M"),n.a.createElement("li",{id:"3M-li",className:"3M"===this.state.timeframe?"active-list":"",onClick:this.handleLowerBoundChange},"3M"),n.a.createElement("li",{id:"ALL-li",className:"ALL"===this.state.timeframe?"active-list":"",onClick:this.handleLowerBoundChange},"ALL")),n.a.createElement("hr",null)),n.a.createElement("div",{className:"control-row two"},n.a.createElement("button",{onClick:this.handleClick,id:"buyButton",className:"button"},"Buy"),n.a.createElement("div",{id:"slideContainer"},n.a.createElement("label",{for:"slider",id:"sliderLabel"},"Speed"),n.a.createElement("input",{onChange:this.handleSlide,type:"range",min:"0",max:"1",step:"0.1",value:this.state.speed,id:"slider"})),n.a.createElement("button",{onClick:this.handleClick,id:"sellButton",className:"button"},"Sell")))))}}]),a}(n.a.Component));var O=function(e){Object(o.a)(a,e);var t=Object(d.a)(a);function a(){return Object(l.a)(this,a),t.call(this)}return Object(c.a)(a,[{key:"render",value:function(){return n.a.createElement("div",{className:"App"},n.a.createElement("header",{className:"App-header"}),n.a.createElement(m.c,null,n.a.createElement(m.a,{exact:!0,path:"/"},n.a.createElement(p,null)),n.a.createElement(m.a,{path:"/game"},n.a.createElement(F,null))))}}]),a}(n.a.Component);Boolean("localhost"===window.location.hostname||"[::1]"===window.location.hostname||window.location.hostname.match(/^127(?:\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}$/));s.a.render(n.a.createElement(u.a,null,n.a.createElement(n.a.StrictMode,null,n.a.createElement(O,null))),document.getElementById("root")),"serviceWorker"in navigator&&navigator.serviceWorker.ready.then((function(e){e.unregister()})).catch((function(e){console.error(e.message)}))}},[[54,1,2]]]);
//# sourceMappingURL=main.8d04e487.chunk.js.map