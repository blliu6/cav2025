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
        model = rational_SOS(f, 1, 0, mono, rQ)
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


#H4
# third
## Multiplier-2
P0 = 293*X[1]^2//1000 - 453*X[1]*X[2]//5000 + 497*X[1]//100000 + 211*X[2]^2//500 + 59*X[2]//200 + 11//40
M0 = [1, X[2], X[1]]
Q0 = [0.275356231897561 0.14792616814104606 0.0023592136729990725;
 0.14792616814104606 0.4227661400952595 -0.04527015099904592;
 0.0023592136729990725 -0.04527015099904592 0.2936168563490378;
]
solve(n, X, P0, Q0, M0, true)
## Total Decomposition
P1 = 257*X[1]^4//3125 - 1031*X[1]^3*X[2]//5000 - 4721*X[1]^3//10000 + 2987*X[1]^2*X[2]^2//10000 - 3213*X[1]^2*X[2]//10000 + 1671*X[1]^2//1000 + 236*X[1]*X[2]^3//625 - 1663*X[1]*X[2]^2//10000 + 341*X[1]*X[2]//1000 - 232*X[1]//125 + 3739*X[2]^4//5000 - 6883*X[2]^3//10000 + 304*X[2]^2//125 - 1523*X[2]//1000 + 639//250
M1 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2]
Q1 = [2.556151527416023 -0.7614841373926529 -0.9280259731426119 0.5441933656545608 0.01892492449676344 0.08119752033249454;
 -0.7614841373926529 1.3429496585694725 0.15310409612405806 -0.34353766909187583 -0.008626926376012352 -0.04128424579877885;
 -0.9280259731426119 0.15310409612405806 1.5080440647112552 -0.07492706453399281 -0.1204114796903032 -0.23610613553377904;
 0.5441933656545608 -0.34353766909187583 -0.07492706453399281 0.7474182617881586 0.18882285038152366 -0.08296309025240853;
 0.01892492449676344 -0.008626926376012352 -0.1204114796903032 0.18882285038152366 0.465498683243702 -0.10286044443929179;
 0.08119752033249454 -0.04128424579877885 -0.23610613553377904 -0.08296309025240853 -0.10286044443929179 0.08187529095852775;
]
solve(n, X, P1, Q1, M1, true)
## Multiplier-1
P2 = 353*X[1]^2//5000 + 103*X[1]*X[2]//20000 + 797*X[1]//10000 + 113*X[2]^2//1000 + 33*X[2]//625 + 3//50
M2 = [1, X[2], X[1]]
Q2 = [0.05974622604775363 0.026288290931171764 0.03966524444002531;
 0.026288290931171764 0.11267311261349931 0.002474227442268029;
 0.03966524444002531 0.002474227442268029 0.07083704141864641;
]
solve(n, X, P2, Q2, M2, true)
# second
## Multiplier-2
P3 = 903*X[1]^2//1000 - 841*X[1]*X[2]//10000 + 341*X[1]//5000 + 953*X[2]^2//1000 + 309*X[2]//1000 + 87//125
M3 = [1, X[2], X[1]]
Q3 = [0.696329045633362 0.15472959449806326 0.03433987773550926;
 0.15472959449806326 0.9530646261382575 -0.042074511309781286;
 0.03433987773550926 -0.042074511309781286 0.9035591453222014;
]
solve(n, X, P3, Q3, M3, true)
## Total Decomposition
P4 = 513*X[1]^4//2000 + 717*X[1]^3*X[2]//2000 - 3603*X[1]^3//5000 + 1183*X[1]^2*X[2]^2//1000 - 5463*X[1]^2*X[2]//10000 + 1061*X[1]^2//500 - 2341*X[1]*X[2]^3//100000 - 947*X[1]*X[2]^2//1250 - 761*X[1]*X[2]//2500 - 259*X[1]//12500 + 987*X[2]^4//1000 - 737*X[2]^3//2000 + 2877*X[2]^2//1000 - 733*X[2]//1250 + 643//250
M4 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2]
Q4 = [2.5722037967670697 -0.2920209383134689 -0.011026049866058717 0.6320504844005717 -0.0541390788850555 0.07834557788634804;
 -0.2920209383134689 1.6111131704538668 -0.09744225992765287 -0.18451228666809957 -0.19885865921418497 -0.02095829058513375;
 -0.011026049866058717 -0.09744225992765287 1.9672413755454026 -0.18064962094652515 -0.25136474322064123 -0.3599952044134609;
 0.6320504844005717 -0.18451228666809957 -0.18064962094652515 0.9868650012327883 -0.01197688862587033 0.050029434663052805;
 -0.0541390788850555 -0.19885865921418497 -0.25136474322064123 -0.01197688862587033 1.0834815253445722 0.17915234329259555;
 0.07834557788634804 -0.02095829058513375 -0.3599952044134609 0.050029434663052805 0.17915234329259555 0.25665260011539803;
]
solve(n, X, P4, Q4, M4, true)
## Multiplier-1
P5 = 933*X[1]^2//10000 + 171*X[1]*X[2]//10000 + 267*X[1]//10000 + 111*X[2]^2//1000 + 113*X[2]//5000 + 47//500
M5 = [1, X[2], X[1]]
Q5 = [0.0939287528604007 0.011276227890443939 0.013316963841433998;
 0.011276227890443939 0.11177755885156525 0.008502753556122813;
 0.013316963841433998 0.008502753556122813 0.09300468128975972;
]
solve(n, X, P5, Q5, M5, true)
# seventh
## Multiplier-2
P6 = 331*X[1]^2//1000 - 61*X[1]*X[2]//1250 - 159*X[1]//25000 + 367*X[2]^2//1000 + 223*X[2]//1000 + 53//200
M6 = [1, X[2], X[1]]
Q6 = [0.26453087992482655 0.11109364704252404 -0.003095145210060237;
 0.11109364704252404 0.36681479929470845 -0.024401031588458084;
 -0.003095145210060237 -0.024401031588458084 0.33036915359264735;
]
solve(n, X, P6, Q6, M6, true)
## Total Decomposition
P7 = 1163*X[1]^4//50000 - 843*X[1]^3*X[2]//50000 - 2327*X[1]^3//10000 + 773*X[1]^2*X[2]^2//2000 - 2653*X[1]^2*X[2]//5000 + 9079*X[1]^2//10000 - 4899*X[1]*X[2]^3//100000 - 4563*X[1]*X[2]^2//10000 - 77*X[1]*X[2]//200 - 1123*X[1]//10000 + 733*X[2]^4//2000 - 343*X[2]^3//625 + 1531*X[2]^2//1000 - 4781*X[2]//10000 + 1811//1000
M7 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2]
Q7 = [1.8111547013140188 -0.23873727095812175 -0.05684927890979541 0.2955693994272359 -0.26023306963642445 -0.005653851139497061;
 -0.23873727095812175 0.9399857628845388 0.06856507989974477 -0.27406189657966035 -0.10605491624072033 -0.0070613290222628616;
 -0.05684927890979541 0.06856507989974477 0.9187908243079327 -0.12281696137073753 -0.2574646620619256 -0.11602692900660862;
 0.2955693994272359 -0.27406189657966035 -0.12281696137073753 0.3668147992947137 -0.02440103158845913 0.012316442518777037;
 -0.26023306963642445 -0.10605491624072033 -0.2574646620619256 -0.02440103158845913 0.36114004841884734 -0.008582757654519766;
 -0.005653851139497061 -0.0070613290222628616 -0.11602692900660862 0.012316442518777037 -0.008582757654519766 0.02365810000043901;
]
solve(n, X, P7, Q7, M7, true)
## Multiplier-1
P8 = 47*X[1]^2//2000 - 43*X[1]*X[2]//2500 + 231*X[1]//50000 + 139*X[2]^2//2500 - 709*X[2]//100000 + 11//500
M8 = [1, X[2], X[1]]
Q8 = [0.021554064376342785 -0.003465952428707962 0.00226357094702958;
 -0.003465952428707962 0.05540377987332022 -0.008582757654547003;
 0.00226357094702958 -0.008582757654547003 0.023658100000673775;
]
solve(n, X, P8, Q8, M8, true)
# eighth
## Total Decomposition
P9 = 10247*X[1]^4//10000 + 16159*X[1]^3*X[2]//100000 - 15911*X[1]^3//2000 + 1623*X[1]^2*X[2]^2//1000 - 20989*X[1]^2*X[2]//5000 + 7427*X[1]^2//500 + 3289*X[1]*X[2]^3//20000 - 47983*X[1]*X[2]^2//10000 + 948*X[1]*X[2]//625 + 73843*X[1]//10000 + 5913*X[2]^4//10000 - 944*X[2]^3//625 + 7001*X[2]^2//500 + 16551*X[2]//10000 + 10609//1000
M9 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2]
Q9 = [10.609452568907834 0.8312102294380216 3.6921177150981928 0.7260186596718251 -1.2398405572203794 -1.2114128426663748;
 0.8312102294380216 12.559756129875122 2.002710048929989 -0.7589397385097612 -2.2668270715895074 -0.6484643281293733;
 3.6921177150981928 2.002710048929989 17.272785566114454 -0.1205286771109497 -1.4564577523496405 -3.978609058394516;
 0.7260186596718251 -0.7589397385097612 -0.1205286771109497 0.5924245222332548 0.08205650187045996 -0.028500622119111817;
 -1.2398405572203794 -2.2668270715895074 -1.4564577523496405 0.08205650187045996 1.6741850322499565 0.0820564920815136;
 -1.2114128426663748 -0.6484643281293733 -3.978609058394516 -0.028500622119111817 0.0820564920815136 1.0247592142281827;
]
solve(n, X, P9, Q9, M9, true)
## Multiplier-1
P10 = 51*X[1]^2//50 + 41*X[1]*X[2]//250 + 377*X[1]//500 + 74*X[2]^2//125 + 259*X[2]//1000 + 161//250
M10 = [1, X[2], X[1]]
Q10 = [0.6437653050458482 0.12969692344518255 0.3766176020004821;
 0.12969692344518255 0.5924244516088074 0.08205649207609785;
 0.3766176020004821 0.08205649207609785 1.0247592142672641;
]
solve(n, X, P10, Q10, M10, true)
# first
## Total Decomposition
P11 = 5571*X[1]^4//100000 - 253*X[1]^3*X[2]//40000 - 5903*X[1]^3//10000 + 1047*X[1]^2*X[2]^2//10000 + 1743*X[1]^2*X[2]//50000 + 1573*X[1]^2//1000 - 3011*X[1]*X[2]^3//500000 - 5079*X[1]*X[2]^2//10000 + 601*X[1]*X[2]//5000 + 273*X[1]//625 + 2419*X[2]^4//50000 - 2797*X[2]^3//100000 + 817*X[2]^2//500 + 107*X[2]//625 + 1241//1000
M11 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2]
Q11 = [1.2413079387830388 0.08534267933577581 0.21836783590959827 0.07557583517023217 -0.0042772862470180405 -0.0729835886511564;
 0.08534267933577581 1.4829289181431324 0.06376646655117894 -0.014561885739087226 -0.23789534896853673 -0.011198710344728304;
 0.21836783590959827 0.06376646655117894 1.7192576537749131 -0.015593478481948015 0.027769116023314153 -0.2950084406588157;
 0.07557583517023217 -0.014561885739087226 -0.015593478481948015 0.04866601312093084 -0.0029955877616093087 -0.006122894449832065;
 -0.0042772862470180405 -0.23789534896853673 0.027769116023314153 -0.0029955877616093087 0.11685451562556419 -0.0029955881187124136;
 -0.0729835886511564 -0.011198710344728304 -0.2950084406588157 -0.006122894449832065 -0.0029955881187124136 0.055942714487760725;
]
solve(n, X, P11, Q11, M11, true)
## Multiplier-1
P12 = 281*X[1]^2//5000 - 603*X[1]*X[2]//100000 + 127*X[1]//5000 + 49*X[2]^2//1000 - 3*X[2]//625 + 1//25
M12 = [1, X[2], X[1]]
Q12 = [0.03983209900585626 -0.0023953824626717036 0.012676450749876186;
 -0.0023953824626717036 0.04866601312092991 -0.00299558776161067;
 0.012676450749876186 -0.00299558776161067 0.055942707818857886;
]
solve(n, X, P12, Q12, M12, true)
# fifth
## Multiplier-2
P13 = 371*X[1]^2//1000 - 109*X[1]*X[2]//2500 - 79*X[1]//2000 + 329*X[2]^2//1000 + 189*X[2]//1000 + 267//1000
M13 = [1, X[2], X[1]]
Q13 = [0.2672206670623157 0.09461774164457702 -0.01974179889459057;
 0.09461774164457702 0.3290085536403067 -0.0217743379621638;
 -0.01974179889459057 -0.0217743379621638 0.37142707225273386;
]
solve(n, X, P13, Q13, M13, true)
## Total Decomposition
P14 = 6053*X[1]^4//100000 - 457*X[1]^3*X[2]//2500 - 3159*X[1]^3//10000 + 501*X[1]^2*X[2]^2//2500 - 136*X[1]^2*X[2]//3125 + 8627*X[1]^2//10000 + 119*X[1]*X[2]^3//400 + 2353*X[1]*X[2]^2//10000 - 81*X[1]*X[2]//625 - 303*X[1]//1000 + 3203*X[2]^4//5000 - 2643*X[2]^3//10000 + 1511*X[2]^2//1000 - 3697*X[2]//10000 + 287//250
M14 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2]
Q14 = [1.1482071701079803 -0.18529046842670846 -0.15128317976487302 0.3505190369566044 -0.012792436692262897 -0.015147721599079903;
 -0.18529046842670846 0.8112828165567948 -0.052117145273524776 -0.13268251422628954 0.029893705863896104 -0.016252304571007445;
 -0.15128317976487302 -0.052117145273524776 0.8922145234815508 0.0878942987683801 -0.005332591675931495 -0.15821831843494555;
 0.3505190369566044 -0.13268251422628954 0.0878942987683801 0.6408821290844218 0.14889189891063026 -0.0867996637992187;
 -0.012792436692262897 0.029893705863896104 -0.005332591675931495 0.14889189891063026 0.3741690629290248 -0.09133046833059627;
 -0.015147721599079903 -0.016252304571007445 -0.15821831843494555 -0.0867996637992187 -0.09133046833059627 0.06078936884102495;
]
solve(n, X, P14, Q14, M14, true)
## Multiplier-1
P15 = 249*X[1]^2//5000 + 29*X[1]*X[2]//2500 + 23*X[1]//500 + 41*X[2]^2//625 + 89*X[2]//5000 + 49//1000
M15 = [1, X[2], X[1]]
Q15 = [0.04879390368347981 0.00889292674502512 0.02290279270819259;
 0.00889292674502512 0.06534472504113588 0.005740458420900225;
 0.02290279270819259 0.005740458420900225 0.05024116026322397;
]
solve(n, X, P15, Q15, M15, true)
# fourth
## Multiplier-2
P16 = 671*X[1]^2//500 - 1343*X[1]*X[2]//5000 - 1257*X[1]//10000 + 1473*X[2]^2//1000 + 9761*X[2]//10000 + 1143//1000
M16 = [1, X[2], X[1]]
Q16 = [1.1434385967821192 0.4883914034823358 -0.06322865478978208;
 0.4883914034823358 1.4735193900554406 -0.13493096805780438;
 -0.06322865478978208 -0.13493096805780438 1.3417332634769155;
]
solve(n, X, P16, Q16, M16, true)
## Total Decomposition
P17 = 4737*X[1]^4//10000 + 6807*X[1]^3*X[2]//10000 - 301*X[1]^3//250 + 511*X[1]^2*X[2]^2//250 - 1579*X[1]^2*X[2]//1000 + 49*X[1]^2//10 + 3599*X[1]*X[2]^3//50000 - 369*X[1]*X[2]^2//250 - 1913*X[1]*X[2]//10000 - 1351*X[1]//10000 + 13*X[2]^4//8 - 153*X[2]^3//100 + 2859*X[2]^2//500 - 7*X[2]//5 + 6139//1000
M17 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2]
Q17 = [6.138790331945795 -0.700069141787681 -0.06611391019901648 1.234551881664956 -0.04287966071457502 0.3655477642190697;
 -0.700069141787681 3.249582613307224 -0.05170979347256856 -0.7658292264042992 -0.42085578119619266 -0.09889773416385565;
 -0.06611391019901648 -0.05170979347256856 4.1623515918654155 -0.31576138614506466 -0.6890087355057357 -0.6016377996607403;
 1.234551881664956 -0.7658292264042992 -0.31576138614506466 1.6249425171250687 0.035902339741962774 0.1009044857348333;
 -0.04287966071457502 -0.42085578119619266 -0.6890087355057357 0.035902339741962774 1.8415431799052528 0.34041943824142;
 0.3655477642190697 -0.09889773416385565 -0.6016377996607403 0.1009044857348333 0.34041943824142 0.47300429099829755;
]
solve(n, X, P17, Q17, M17, true)
## Multiplier-1
P18 = 161*X[1]^2//1000 - 813*X[1]*X[2]//100000 + 33*X[1]//500 + 211*X[2]^2//1000 + 71*X[2]//25000 + 129//1000
M18 = [1, X[2], X[1]]
Q18 = [0.1286580137063582 0.0013962361893638247 0.032975155086019003;
 0.0013962361893638247 0.2113487082149453 -0.0039995874727472;
 0.032975155086019003 -0.0039995874727472 0.1609202552209801;
]
solve(n, X, P18, Q18, M18, true)
# sixth
## Multiplier-2
P19 = 51*X[1]^2//200 - 67*X[1]*X[2]//2000 - 247*X[1]//50000 + 7*X[2]^2//25 + 21*X[2]//125 + 103//500
M19 = [1, X[2], X[1]]
Q19 = [0.20555507137177784 0.0836451469506429 -0.0024809312970562013;
 0.0836451469506429 0.27993961123456407 -0.016777221002095492;
 -0.0024809312970562013 -0.016777221002095492 0.25511859115271635;
]
solve(n, X, P19, Q19, M19, true)
## Total Decomposition
P20 = 1323*X[1]^4//100000 - 641*X[1]^3*X[2]//50000 - 1343*X[1]^3//10000 + 2947*X[1]^2*X[2]^2//10000 - 3811*X[1]^2*X[2]//10000 + 5907*X[1]^2//10000 - 1679*X[1]*X[2]^3//50000 - 42*X[1]*X[2]^2//125 - 2913*X[1]*X[2]//10000 - 1877*X[1]//10000 + 2799*X[2]^4//10000 - 1947*X[2]^3//5000 + 1103*X[2]^2//1000 - 443*X[2]//1250 + 579//500
M20 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2]
Q20 = [1.1576647516208785 -0.17732261840331043 -0.09402018512757311 0.20882930110260314 -0.17229348673890377 0.006984046401878198;
 -0.17732261840331043 0.6863070185133775 0.02662332829420273 -0.19489476756166418 -0.07569551621463406 -0.0016477201376760646;
 -0.09402018512757311 0.02662332829420273 0.577354725780738 -0.0921858322147864 -0.18902623688613795 -0.06695514775856574;
 0.20882930110260314 -0.19489476756166418 -0.0921858322147864 0.27993961123456285 -0.016777221002094667 0.009570631588664637;
 -0.17229348673890377 -0.07569551621463406 -0.18902623688613795 -0.016777221002094667 0.2757347453926716 -0.006548587645843359;
 0.006984046401878198 -0.0016477201376760646 -0.06695514775856574 0.009570631588664637 -0.006548587645843359 0.013409147071586204;
]
solve(n, X, P20, Q20, M20, true)
## Multiplier-1
P21 = 131*X[1]^2//10000 - 13*X[1]*X[2]//1000 + 19*X[1]//100000 + 393*X[2]^2//10000 - 471*X[2]//100000 + 19//1000
M21 = [1, X[2], X[1]]
Q21 = [0.018747633067387626 -0.0023168340703355115 9.058759217978567e-05;
 -0.0023168340703355115 0.03975741741823223 -0.006548587645864978;
 9.058759217978567e-05 -0.006548587645864978 0.013409147071178079;
]
solve(n, X, P21, Q21, M21, true)