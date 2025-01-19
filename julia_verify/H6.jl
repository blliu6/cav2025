include("utils/util.jl")
include("utils/newtonUpdate.jl")
include("utils/rationalSOS.jl")

X = nothing
eig_tol = 1e-6
tol = 1e-50

SOS_time = 0
Newton_time = 0
function solve(n, X, f, Q, mono,newton)
    global SOS_time, Newton_time
    Q = round.(Q, digits=2)
    rQ = symplify_rational.(Q)

    model = nothing
    if !newton
        SOS_time+=@elapsed model = rational_SOS(f, 1, 0, mono, rQ)
    else
        Newton_time+=@elapsed update_Q = newton_refine_update(f, mono, Q, eig_tol, 90, tol, X)
        Q_res = symplify_rational.(update_Q*update_Q')
        SOS_time+=@elapsed model = rational_SOS(f, 1, 0, mono, Q_res)
    end
    
    SOSrational_time, coeffs, terms = model[2:end]
    
    @show expand(f-(coeffs' * terms.^2))

    data = Dict(
        "n" => n,
        "expression" => f,
        "coeff" => join([string(c) for c in coeffs], ","),
        "term" => join([string(t) for t in terms], ","),
        "SOSrational_time" => SOSrational_time,
        "SOS_expression" => string(coeffs' * terms.^2)
    )

    # save("src/output/data.json", data)    
end

n = 2
@Symbolics.variables X[1:n]


#H6
# third
## Multiplier-2
P0 = 9723*X[1]^2//10000 - 3717*X[1]*X[2]//10000 + 3789*X[1]//10000 + 4587*X[2]^2//5000 - 211*X[2]//2000 + 1199//1000
M0 = [1, X[2], X[1]]
Q0 = [1.1991113358827115 -0.05322344725298629 0.18956151793019538;
 -0.05322344725298629 0.9176997001061585 -0.1858615850995971;
 0.18956151793019538 -0.1858615850995971 0.9725665383293397;
]
rational_SOS(P0, 1, 0, M0, Q0)
solve(n, X, P0, Q0, M0, false)
## Total Decomposition
P1 = 207*X[1]^4//100 + 1123*X[1]^3*X[2]//1000 - 1001*X[1]^3//2500 + 3599*X[1]^2*X[2]^2//1000 + 128*X[1]^2*X[2]//125 + 503*X[1]^2//500 + 2781*X[1]*X[2]^3//100000 + 487*X[1]*X[2]^2//2000 - 1403*X[1]*X[2]//1000 + 533*X[1]//1250 + 1131*X[2]^4//1000 + 693*X[2]^3//500 + 2053*X[2]^2//2500 - 2009*X[2]//10000 + 881//500
M1 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2]
Q1 = [1.7620053518566212 -0.10082814134436281 0.21273821599884946 -0.4049625714152439 -0.4379112205798783 -0.34307734467386763;
 -0.10082814134436281 1.631598799779491 -0.26281175652329763 0.6932048136196488 0.21262312881936707 0.4576752989428;
 0.21273821599884946 -0.26281175652329763 1.693626408162063 -0.089497614037922 0.054873148931323806 -0.20076002083670555;
 -0.4049625714152439 0.6932048136196488 -0.089497614037922 1.1305216439346573 0.013886857947538031 0.4534121168697118;
 -0.4379112205798783 0.21262312881936707 0.054873148931323806 0.013886857947538031 2.693038446309791 0.5604437598960266;
 -0.34307734467386763 0.4576752989428 -0.20076002083670555 0.4534121168697118 0.5604437598960266 2.069360343405747;
]
solve(n, X, P1, Q1, M1, false)
## Multiplier-1
P2 = 1069*X[1]^2//1000 - 9289*X[1]*X[2]//1000000 + 1191*X[2]^2//1000 - 933*X[2]//2000 + 1431//1000
M2 = [1, X[2], X[1]]
Q2 = [1.4314168896705477 -0.23384622111528688 0.0001508371843145023;
 -0.23384622111528688 1.190888959644698 -0.004430785949699422;
 0.0001508371843145023 -0.004430785949699422 1.0691358313540913;
]
solve(n, X, P2, Q2, M2, false)
# second
## Multiplier-2
P3 = 6667*X[1]^2//10000 + 1929*X[1]*X[2]//5000 + 403*X[1]//500 + 1707*X[2]^2//1000 - 129*X[2]//100 + 227//250
M3 = [1, X[2], X[1]]
Q3 = [0.9079732676982051 -0.6429102032234931 0.4032043934195848;
 -0.6429102032234931 1.7071793378463505 0.19260188320668464;
 0.4032043934195848 0.19260188320668464 0.6666975296373954;
]
solve(n, X, P3, Q3, M3, false)
## Total Decomposition
P4 = 129*X[1]^4//125 - 121*X[1]^3*X[2]//6250 + 2231*X[1]^3//5000 + 2559*X[1]^2*X[2]^2//1000 - 51*X[1]^2*X[2]//250 + 4027*X[1]^2//5000 + 303*X[1]*X[2]^3//2500 - 3477*X[1]*X[2]^2//5000 + 391*X[1]*X[2]//250 + 7063*X[1]//10000 + 1719*X[2]^4//1000 - 9031*X[2]^3//10000 + 2337*X[2]^2//1000 - 177*X[2]//125 + 327//200
M4 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2]
Q4 = [1.6348376891950331 -0.7080806413840982 0.3533216502400373 0.09381304875145659 0.6047389325113562 -0.25998494012619927;
 -0.7080806413840982 2.1505528041041924 0.17729078458236786 -0.45175592613859994 -0.40037463963646575 -0.009067476651430758;
 0.3533216502400373 0.17729078458236786 1.3252906934252728 0.05273563398793159 -0.09321941068629411 0.223364006901198;
 0.09381304875145659 -0.45175592613859994 0.05273563398793159 1.7197390948560718 0.06081030518990107 0.155419217298048;
 0.6047389325113562 -0.40037463963646575 -0.09321941068629411 0.06081030518990107 2.2485747420357 -0.009248146082886237;
 -0.25998494012619927 -0.009067476651430758 0.223364006901198 0.155419217298048 -0.009248146082886237 1.0327262345546897;
]
solve(n, X, P4, Q4, M4, false)
## Multiplier-1
P5 = 543*X[1]^2//500 + 7167*X[1]*X[2]//100000 + 461*X[1]//1000 + 1067*X[2]^2//1000 - 343*X[2]//1000 + 461//500
M5 = [1, X[2], X[1]]
Q5 = [0.9222851477194982 -0.17108587431227176 0.23085299629939984;
 -0.17108587431227176 1.0669567717345412 0.03621950957049668;
 0.23085299629939984 0.03621950957049668 1.0857976796872761;
]
solve(n, X, P5, Q5, M5, false)
# seventh
## Multiplier-2
P6 = 677*X[1]^2//10000 + 169*X[1]//1000 + 99*X[2]^2//1000 + 191//500
M6 = [1, X[2], X[1]]
Q6 = [0.3822195322025234 9.867377504687403e-19 0.08434376890454709;
 9.867377504687403e-19 0.09866704282245056 5.924239787002033e-19;
 0.08434376890454709 5.924239787002033e-19 0.06742630907438707;
]
solve(n, X, P6, Q6, M6, false)
## Total Decomposition
P7 = 141*X[1]^4//1000 + 39*X[1]^3//125 + 37*X[1]^2*X[2]^2//200 + 29*X[1]^2//1000 + 101*X[1]*X[2]^2//250 - 201*X[1]//1000 + 123*X[2]^4//1250 - 61*X[2]^2//5000 + 277//500
M7 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2]
Q7 = [0.554062249721291 -3.525597062603505e-18 -0.10022556410103058 -0.11412068988691441 -1.7021113320529122e-18 -0.14474469294390474;
 -3.525597062603505e-18 0.21579279346473015 9.238322553198602e-17 3.0463208819472085e-17 0.13242968492601243 7.377439848791186e-17;
 -0.10022556410103058 9.238322553198602e-17 0.3178557904005155 0.06915017112753705 6.123694089543916e-17 0.15582744683108946;
 -0.11412068988691441 3.0463208819472085e-17 0.06915017112753705 0.09866704282245019 2.1122446606516635e-17 0.023552858433961224;
 -1.7021113320529122e-18 0.13242968492601243 6.123694089543916e-17 2.1122446606516635e-17 0.13755667935546884 4.6363440874371295e-17;
 -0.14474469294390474 7.377439848791186e-17 0.15582744683108946 0.023552858433961224 4.6363440874371295e-17 0.14059694419553775;
]
solve(n, X, P7, Q7, M7, false)
## Multiplier-1
P8 = 141*X[1]^2//1000 + 303*X[1]//10000 + 117*X[2]^2//1000 + 237//1000
M8 = [1, X[2], X[1]]
Q8 = [0.23714952471170364 5.082814732834739e-18 0.01523051583009659;
 5.082814732834739e-18 0.11723610034354542 7.384667417190556e-18;
 0.01523051583009659 7.384667417190556e-18 0.14059694419553495;
]
solve(n, X, P8, Q8, M8, false)
# eighth
## Total Decomposition
P9 = 14387*X[1]^4//5000 - 12451*X[1]^3*X[2]//50000 - 10999*X[1]^3//2500 + 5777*X[1]^2*X[2]^2//1000 - 20091*X[1]^2*X[2]//5000 + 80041*X[1]^2//10000 - 3111*X[1]*X[2]^3//12500 - 9831*X[1]*X[2]^2//2500 - 603*X[1]*X[2]//125 + 17557*X[1]//10000 + 28943*X[2]^4//10000 - 45491*X[2]^3//10000 + 28647*X[2]^2//5000 + 2343*X[2]//4000 + 1591//125
M9 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2]
Q9 = [12.728174082815 0.2956524109575685 0.879934252057112 -0.5009681572442135 -2.527979158650372 0.24801951821214016;
 0.2956524109575685 6.7230821287906855 0.1210964734918536 -2.27254775790105 -1.2242267226043952 -0.5774096337699299;
 0.879934252057112 0.1210964734918536 7.520313669988944 -0.7391460081772968 -1.4272266348013876 -2.197444660682757;
 -0.5009681572442135 -2.27254775790105 -0.7391460081772968 2.893247471270871 -0.12549585480770165 0.2715350840860555;
 -2.527979158650372 -1.2242267226043952 -1.4272266348013876 -0.12549585480770165 5.226504994617078 -0.12549585480770076;
 0.24801951821214016 -0.5774096337699299 -2.197444660682757 0.2715350840860555 -0.12549585480770076 2.8763276915565483;
]
solve(n, X, P9, Q9, M9, false)
## Multiplier-1
P10 = 719*X[1]^2//250 - 2521*X[1]*X[2]//10000 + 1357*X[1]//1000 + 1447*X[2]^2//500 + 1243*X[2]//1000 + 13//5
M10 = [1, X[2], X[1]]
Q10 = [2.5996698865478414 0.6206997133315911 0.6788830308355471;
 0.6206997133315911 2.8932474712708807 -0.12549585480770173;
 0.6788830308355471 -0.12549585480770173 2.876327691556546;
]
solve(n, X, P10, Q10, M10, false)
# first
## Total Decomposition
P11 = 3341*X[1]^4//1000 - 2717*X[1]^3*X[2]//10000 + 53*X[1]^3//10 + 7083*X[1]^2*X[2]^2//1000 + 2033*X[1]^2*X[2]//1000 + 473*X[1]^2//50 - 2659*X[1]*X[2]^3//10000 + 5573*X[1]*X[2]^2//1000 - 667*X[1]*X[2]//100 - 277*X[1]//100 + 3741*X[2]^4//1000 + 421*X[2]^3//125 + 3007*X[2]^2//500 + 2517*X[2]//500 + 1979//500
M11 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2]
Q11 = [3.9577199580720777 2.5164991160603147 -1.3834412513695322 -0.43907494636573363 -0.9155922681889042 -0.13040156095707592;
 2.5164991160603147 6.891862252706825 -2.4200802278304887 1.685319752001614 1.1904027994046862 -0.059545137983621814;
 -1.3834412513695322 -2.4200802278304887 9.720919585021997 1.5903135129426729 1.078117679235438 2.648760266552401;
 -0.43907494636573363 1.685319752001614 1.5903135129426729 3.739863836633249 -0.133697791238711 0.480863113447297;
 -0.9155922681889042 1.1904027994046862 1.078117679235438 -0.133697791238711 6.118649818081805 -0.13369779123871442;
 -0.13040156095707592 -0.059545137983621814 2.648760266552401 0.480863113447297 -0.13369779123871442 3.340512208360864;
]
solve(n, X, P11, Q11, M11, false)
## Multiplier-1
P12 = 3341*X[1]^2//1000 - 166*X[1]*X[2]//625 - 693*X[1]//500 + 187*X[2]^2//50 - 4107*X[2]//1000 + 199//25
M12 = [1, X[2], X[1]]
Q12 = [7.96001682657949 -2.0545440846139145 -0.6917519417907666;
 -2.0545440846139145 3.7398638366332486 -0.13369779123871683;
 -0.6917519417907666 -0.13369779123871683 3.3405122083608663;
]
solve(n, X, P12, Q12, M12, false)
# fifth
## Multiplier-2
P13 = 1393*X[1]^2//1000 - 1001*X[1]*X[2]//50000 - 9233*X[1]//10000 + 1127*X[2]^2//1000 + 4407*X[2]//500000 + 547//250
M13 = [1, X[2], X[1]]
Q13 = [2.1884507796985204 0.005136479765712506 -0.46173820961870365;
 0.005136479765712506 1.1267493991119502 -0.010325392038338528;
 -0.46173820961870365 -0.010325392038338528 1.393529814772968;
]
solve(n, X, P13, Q13, M13, false)
## Total Decomposition
P14 = 437*X[1]^4//200 - 3467*X[1]^3*X[2]//10000 + 1933*X[1]^3//1000 + 419*X[1]^2*X[2]^2//100 - 47*X[1]^2*X[2]//25 + 977*X[1]^2//500 + 517*X[1]*X[2]^3//2000 + 2073*X[1]*X[2]^2//1000 + 1699*X[1]*X[2]//10000 - 1079*X[1]//1000 + 709*X[2]^4//500 + 1593*X[2]^3//1000 + 7851*X[2]^2//10000 + 1883*X[2]//10000 + 509//200
M14 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2]
Q14 = [2.544614591501183 0.09389804007584664 -0.5398228409062277 -0.6272590834605165 0.11644200090031931 -0.4160477583606083;
 0.09389804007584664 2.042146954609652 -0.0330013438012013 0.7965831633529876 0.6227489001436588 -0.18609421100574927;
 -0.5398228409062277 -0.0330013438012013 2.788615299746581 0.41396801021816343 -0.7536976869194981 0.966750444183923;
 -0.6272590834605165 0.7965831633529876 0.41396801021816343 1.4177337323028487 0.1299377096306895 0.35858869523061804;
 0.11644200090031931 0.6227489001436588 -0.7536976869194981 0.1299377096306895 3.47199514223651 -0.17276430237097026;
 -0.4160477583606083 -0.18609421100574927 0.966750444183923 0.35858869523061804 -0.17276430237097026 2.184567695010134;
]
solve(n, X, P14, Q14, M14, false)
## Multiplier-1
P15 = 267*X[1]^2//200 + 449*X[1]*X[2]//1250 - 3611*X[1]//10000 + 1229*X[2]^2//1000 + 299*X[2]//1000 + 1941//1000
M15 = [1, X[2], X[1]]
Q15 = [1.9410806691679114 0.1490318816572262 -0.17999752333190475;
 0.1490318816572262 1.2281792176366528 0.1794659212270346;
 -0.17999752333190475 0.1794659212270346 1.3342246248969438;
]
solve(n, X, P15, Q15, M15, false)
# fourth
## Multiplier-2
P16 = 647*X[1]^2//200 + 3177*X[1]*X[2]//5000 + 981*X[1]//500 + 989*X[2]^2//500 + 1293*X[2]//1000 + 93//40
M16 = [1, X[2], X[1]]
Q16 = [2.325051168299361 0.6473655796325901 0.9806217150703497;
 0.6473655796325901 1.9785796277548995 0.31780998109157244;
 0.9806217150703497 0.31780998109157244 3.2347576980656996;
]
solve(n, X, P16, Q16, M16, false)
## Total Decomposition
P17 = 724*X[1]^4//125 - 1147*X[1]^3*X[2]//1000 + 397*X[1]^3//2500 + 1889*X[1]^2*X[2]^2//200 + 279*X[1]^2*X[2]//125 + 19*X[1]^2//5 - 587*X[1]*X[2]^3//250 - 5213*X[1]*X[2]^2//10000 + 1203*X[1]*X[2]//1000 + 203*X[1]//125 + 2157*X[2]^4//1000 + 3719*X[2]^3//1000 + 689*X[2]^2//500 + 6149*X[2]//10000 + 1147//250
M17 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2]
Q17 = [4.588298903271037 0.30560165574874326 0.8129702250733214 -1.2621064487930567 0.5783178960932807 -0.733828584227058;
 0.30560165574874326 3.901848280030649 0.022578290123872215 1.8592183656696837 -0.25196505637939237 0.6744123102323804;
 0.8129702250733214 0.022578290123872215 5.268206462664531 -0.009727119320874864 0.442088098560772 0.08109659736056538;
 -1.2621064487930567 1.8592183656696837 -0.009727119320874864 2.15504104610146 -1.1734698562336157 0.761357719249024;
 0.5783178960932807 -0.25196505637939237 0.442088098560772 -1.1734698562336157 7.923438856666186 -0.5707128620057501;
 -0.733828584227058 0.6744123102323804 0.08109659736056538 0.761357719249024 -0.5707128620057501 5.792310369629988;
]
solve(n, X, P17, Q17, M17, false)
## Multiplier-1
P18 = 2269*X[1]^2//500 + 1013*X[1]*X[2]//500 - 3041*X[1]//1000 + 689*X[2]^2//100 - 5313*X[2]//1000 + 1118//125
M18 = [1, X[2], X[1]]
Q18 = [8.94427523861895 -2.660809245627967 -1.5214141264132124;
 -2.660809245627967 6.892960890051392 1.0119055324070774;
 -1.5214141264132124 1.0119055324070774 4.538031052771984;
]
solve(n, X, P18, Q18, M18, false)
# sixth
## Multiplier-2
P19 = 321*X[1]^2//5000 - 81*X[1]//500 + 47*X[2]^2//500 + 9//25
M19 = [1, X[2], X[1]]
Q19 = [0.3597424402529853 1.9805726232666516e-18 -0.08095012057944503;
 1.9805726232666516e-18 0.09353959977479287 -1.1980241812894196e-18;
 -0.08095012057944503 -1.1980241812894196e-18 0.06468360747770831;
]
solve(n, X, P19, Q19, M19, false)
## Total Decomposition
P20 = 67*X[1]^4//500 - 3*X[1]^3//10 + 177*X[1]^2*X[2]^2//1000 + 19*X[1]^2//625 - 77*X[1]*X[2]^2//200 + 193*X[1]//1000 + 233*X[2]^4//2500 - 7*X[2]^2//500 + 529//1000
M20 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2]
Q20 = [0.5292821825763445 -7.922286481575139e-18 0.09633245562637453 -0.10958635548128957 -1.301267742002389e-20 -0.13788443684977558;
 -7.922286481575139e-18 0.2047567521164356 5.1602591965989175e-17 1.980581159316424e-18 -0.12599682892842062 9.405873720552273e-18;
 0.09633245562637453 5.1602591965989175e-17 0.30665872675621736 -0.06640540145495422 -2.42400949303924e-17 -0.15012355287725507;
 -0.10958635548128957 1.980581159316424e-18 -0.06640540145495422 0.09353959977479237 -1.1980291086060033e-18 0.022841073912153717;
 -1.301267742002389e-20 -0.12599682892842062 -2.42400949303924e-17 -1.1980291086060033e-18 0.13045356945732875 -4.282260724122701e-18;
 -0.13788443684977558 9.405873720552273e-18 -0.15012355287725507 0.022841073912153717 -4.282260724122701e-18 0.13458448880825416;
]
solve(n, X, P20, Q20, M20, false)
## Multiplier-1
P21 = 27*X[1]^2//200 - 309*X[1]//10000 + 111*X[2]^2//1000 + 227//1000
M21 = [1, X[2], X[1]]
Q21 = [0.22746802669141478 -2.3398747465173078e-17 -0.01553906406901305;
 -2.3398747465173078e-17 0.11145210980393917 -4.282255605145879e-18;
 -0.01553906406901305 -4.282255605145879e-18 0.1345844888082532;
]
solve(n, X, P21, Q21, M21, false)

@show SOS_time Newton_time