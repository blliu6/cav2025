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
    # Q = round.(Q, digits=2)
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
#C4
# init
## Multiplier-2
P0 = 19491*X[1]^2//1000 - 4279*X[1]*X[2]//2500 + 953*X[1]//5000 + 3397*X[2]^2//250 - 61*X[2]//20 + 1083//125
M0 = [1, X[2], X[1]]
Q0 = [8.663954795410554 -1.5240301471459137 0.09893092884276748;
 -1.5240301471459137 13.588184678145954 -0.8507337383402714;
 0.09893092884276748 -0.8507337383402714 19.490233036507338;
]
rational_SOS(P0, 1, 0, M0, Q0)
solve(n, X, P0, Q0, M0, false)
## Total Decomposition
P1 = 577*X[1]^4//20 - 71743*X[1]^3*X[2]//100000 + 3013*X[1]^3//2000 + 929*X[1]^2*X[2]^2//25 - 44799*X[1]^2*X[2]//1000 + 4901*X[1]^2//125 - 2151*X[1]*X[2]^3//1250 - 27883*X[1]*X[2]^2//5000 + 453*X[1]*X[2]//20 - 892*X[1]//25 + 13567*X[2]^4//1000 - 11167*X[2]^3//200 + 11977*X[2]^2//100 - 3193*X[2]//20 + 23231//250
M1 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2]
Q1 = [92.92367540715485 -79.85272902041051 -17.884076464273978 11.750536314485766 -0.5125681748930674 5.195230830232378;
 -79.85272902041051 96.3977980282082 11.912709801981757 -27.96425320498799 -1.9222373339619374 -12.97494382159865;
 -17.884076464273978 11.912709801981757 28.813779694524076 -0.9232448183995096 -9.426999102892895 0.7512148074292769;
 11.750536314485766 -27.96425320498799 -0.9232448183995096 13.58818467815308 -0.8507337383395818 5.658196314743442;
 -0.5125681748930674 -1.9222373339619374 -9.426999102892895 -0.8507337383395818 25.858838945936235 -0.35538632749135646;
 5.195230830232378 -12.97494382159865 0.7512148074292769 5.658196314743442 -0.35538632749135646 28.855371820618323;
]
solve(n, X, P1, Q1, M1, false)
## Multiplier-1
P2 = 3607*X[1]^2//125 - 71219*X[1]*X[2]//100000 - 24573*X[1]//10000 + 17683*X[2]^2//1000 - 19517*X[2]//5000 + 24079//1000
M2 = [1, X[2], X[1]]
Q2 = [24.078969865125142 -1.9599348596242616 -1.2285209996409319;
 -1.9599348596242616 17.684998538925292 -0.3553863274914123;
 -1.2285209996409319 -0.3553863274914123 28.855371820618153;
]
solve(n, X, P2, Q2, M2, false)
# unsafe
## Multiplier-2
P3 = 41821*X[1]^2//10000 - 10403*X[1]*X[2]//2500 - 5297*X[1]//500 + 71223*X[2]^2//10000 - 3839*X[2]//250 + 33527//1000
M3 = [1, X[2], X[1]]
Q3 = [33.52653108881569 -7.661389538030128 -5.307137475756416;
 -7.661389538030128 7.115132435817255 -2.0809426813416185;
 -5.307137475756416 -2.0809426813416185 4.1854308175152;
]
solve(n, X, P3, Q3, M3, false)
## Total Decomposition
P4 = 999*X[1]^4//250 - 58327*X[1]^3*X[2]//10000 - 2991*X[1]^3//250 + 1429*X[1]^2*X[2]^2//125 - 43547*X[1]^2*X[2]//10000 + 21153*X[1]^2//1000 - 20781*X[1]*X[2]^3//5000 - 5077*X[1]*X[2]^2//500 + 56923*X[1]*X[2]//10000 - 123*X[1]//10 + 8893*X[2]^4//1250 - 3589*X[2]^3//250 + 9007*X[2]^2//500 - 6829*X[2]//1000 + 5819//1000
M4 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2]
Q4 = [5.8170224179575944 -3.4108952545011877 -6.150151922219378 1.1727760247958123 0.7804213548858469 1.87358184224899;
 -3.4108952545011877 15.661108012748686 2.063077540217915 -7.180572919014813 -2.459631842719949 -0.728541985584661;
 -6.150151922219378 2.063077540217915 17.400291845280556 -2.603797245025781 -1.450358838267846 -5.980058648290267;
 1.1727760247958123 -7.180572919014813 -2.603797245025781 7.115132435815433 -2.0809426813407588 1.7365247017488947;
 0.7804213548858469 -2.459631842719949 -1.450358838267846 -2.0809426813407588 7.95083717223256 -2.9132600455715627;
 1.87358184224899 -0.728541985584661 -5.980058648290267 1.7365247017488947 -2.9132600455715627 3.99227629756712;
]
solve(n, X, P4, Q4, M4, false)
## Multiplier-1
P5 = 3993*X[1]^2//1000 - 58257*X[1]*X[2]//10000 + 39861*X[1]//10000 + 18093*X[2]^2//2500 - 31533*X[2]//5000 + 11723//1000
M5 = [1, X[2], X[1]]
Q5 = [11.722810965738946 -3.1560306399785314 1.9880915971418056;
 -3.1560306399785314 7.238455760516115 -2.913260045570548;
 1.9880915971418056 -2.913260045570548 3.9922762975732393;
]
solve(n, X, P5, Q5, M5, false)
# Lie
## Multiplier-2
P6 = 10921*X[1]^4//10000 + 17479*X[1]^3*X[2]//10000 - 2453*X[1]^3//1250 + 4361*X[1]^2*X[2]^2//1250 - 42697*X[1]^2*X[2]//10000 + 49093*X[1]^2//500000 + 10479*X[1]*X[2]^3//2500 - 3847*X[1]*X[2]^2//250 + 2686*X[1]*X[2]//125 - 14879*X[1]//1000 + 52979*X[2]^4//10000 - 2381*X[2]^3//125 + 35087*X[2]^2//1000 - 11203*X[2]//250 + 14091//500
M6 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2]
Q6 = [28.18209727485652 -22.40827081434593 -7.451288283177177 3.3498904445502373 3.0488280170298294 -2.7669978854776662;
 -22.40827081434593 28.382203911135885 7.714671498105522 -9.524412805445403 -6.01236714395128 1.19331413847803;
 -7.451288283177177 7.714671498105522 5.630151060063795 -1.687122419701169 -3.3292564479228948 -0.9790559415133272;
 3.3498904445502373 -9.524412805445403 -1.687122419701169 5.301336235325322 2.0956139480550777 0.1931669256996803;
 3.0488280170298294 -6.01236714395128 -3.3292564479228948 2.0956139480550777 3.1046675272707995 0.8727058469958996;
 -2.7669978854776662 1.19331413847803 -0.9790559415133272 0.1931669256996803 0.8727058469958996 1.0925799912318817;
]
solve(n, X, P6, Q6, M6, false)
## Total Decomposition
P7 = 2811*X[1]^6//1250 + 911*X[1]^5*X[2]//250 - 3963*X[1]^5//625 + 43223*X[1]^4*X[2]^2//5000 - 7043*X[1]^4*X[2]//10000 - 1033*X[1]^4//625 + 9279*X[1]^3*X[2]^3//1250 - 7841*X[1]^3*X[2]^2//500 - 37693*X[1]^3*X[2]//10000 + 64957*X[1]^3//10000 + 75443*X[1]^2*X[2]^4//10000 - 41693*X[1]^2*X[2]^3//1000 + 11137*X[1]^2*X[2]^2//200 - 40179*X[1]^2*X[2]//1000 + 2971*X[1]^2//250 + 41851*X[1]*X[2]^5//10000 - 7689*X[1]*X[2]^4//500 + 6899*X[1]*X[2]^3//200 - 827*X[1]*X[2]^2//20 + 7497*X[1]*X[2]//250 - 15083*X[1]//1000 + 53083*X[2]^6//10000 - 9531*X[2]^5//500 + 7801*X[2]^4//250 - 3976*X[2]^3//125 + 16021*X[2]^2//500 - 3529*X[2]//125 + 21067//1000
M7 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2, X[2]^3, X[1]*X[2]^2, X[1]^2*X[2], X[1]^3]
Q7 = [21.066617097585265 -14.106079287097792 -7.5424639451977855 3.0269377044476453 9.923746019119777 -3.4699131334356066 0.45638201584385946 -4.273609674996711 -4.067069521504233 1.1944546735335417;
 -14.106079287097792 25.959045903071317 5.073568968638318 -16.366694162532543 -12.99115265912491 -1.784424231234373 2.6291534758070054 5.274956497257289 4.408444096586954 1.0650433690936207;
 -7.5424639451977855 5.073568968638318 18.837252212020978 -3.4034442704005925 -14.242049048101327 2.0437518444894236 2.103369404929905 4.327072488190885 -2.10412147246295 -4.192473947192793;
 3.0269377044476453 -16.366694162532543 -3.4034442704005925 25.94836210073648 9.874430790256753 3.9179017877452837 -9.524412805441319 -5.9472567729252654 -5.3794371589091154 -1.1748505143535453;
 9.923746019119777 -12.99115265912491 -14.242049048101327 9.874430790256753 30.372999187873795 -0.8370481181072745 -1.7522327907318582 -14.198204617425068 -7.383463250943939 2.4284929031200124;
 -3.4699131334356066 -1.784424231234373 2.0437518444894236 3.9179017877452837 -0.8370481181072745 6.718515289159383 -1.2643470839765756 0.7164071637602598 -2.773608193936861 -3.1666841529240646;
 0.45638201584385946 2.6291534758070054 2.103369404929905 -9.524412805441319 -1.7522327907318582 -1.2643470839765756 5.301336235321781 2.0956139480621494 -0.3186206644013212 0.11425898919900308;
 -4.273609674996711 5.274956497257289 4.327072488190885 -5.9472567729252654 -14.198204617425068 0.7164071637602598 2.0956139480621494 8.183560893377022 3.594336475425254 -0.7574219593899587;
 -4.067069521504233 4.408444096586954 -2.10412147246295 -5.3794371589091154 -7.383463250943939 -2.773608193936861 -0.3186206644013212 3.594336475425254 10.149515553215098 1.819172504715366;
 1.1944546735335417 1.0650433690936207 -4.192473947192793 -1.1748505143535453 2.4284929031200124 -3.1666841529240646 0.11425898919900308 -0.7574219593899587 1.819172504715366 2.249545343604736;
]
solve(n, X, P7, Q7, M7, false)
## Multiplier-1
P8 = 5627*X[1]^4//2500 + 9103*X[1]^3*X[2]//2500 - 15841*X[1]^3//2500 + 15087*X[1]^2*X[2]^2//2000 - 24453*X[1]^2*X[2]//1000 + 3681*X[1]^2//200 + 14169*X[1]*X[2]^3//2500 - 25593*X[1]*X[2]^2//1000 + 6091*X[1]*X[2]//125 - 35037*X[1]//1000 + 4053*X[2]^4//1000 - 17977*X[2]^3//1000 + 37473*X[2]^2//1000 - 23811*X[2]//500 + 29541//1000
M8 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2]
Q8 = [29.541180395434488 -23.821175743875447 -17.532170924628232 4.130633574774935 8.740904755381166 0.3505843705734688;
 -23.821175743875447 29.21731079910302 15.64026109894258 -8.988918567867852 -9.823051568165422 -2.341542978325822;
 -17.532170924628232 15.64026109894258 17.713074111491892 -2.977815495513287 -9.881987152562845 -3.1666841529152028;
 4.130633574774935 -8.988918567867852 -2.977815495513287 4.055318186111208 2.8358896176026502 0.6577586789222136;
 8.740904755381166 -9.823051568165422 -9.881987152562845 2.8358896176026502 6.2265742853644275 1.819172504718287;
 0.3505843705734688 -2.341542978325822 -3.1666841529152028 0.6577586789222136 1.819172504718287 2.2495453437378337;
]
solve(n, X, P8, Q8, M8, false)

@show SOS_time Newton_time