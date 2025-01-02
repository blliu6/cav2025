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

#NF1
# init
## Total Decomposition
P0 = 5217*X[1]^4//125 - 44057*X[1]^3*X[2]//5000 + 12881*X[1]^3//500 + 18691*X[1]^2*X[2]^2//200 - 8117*X[1]^2*X[2]//250 + 24249*X[1]^2//1000 - 42167*X[1]*X[2]^3//10000 + 11351*X[1]*X[2]^2//250 + 70597*X[1]*X[2]//1000 + 32247*X[1]//1000 + 39413*X[2]^4//1000 - 46507*X[2]^3//1000 + 12867*X[2]^2//500 - 18289*X[2]//2500 + 10607//200
M0 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2]
Q0 = [53.03494594781823 -3.662137932239019 16.127438562063933 -13.343643753797938 28.189042396975776 -13.053565290872937;
 -3.662137932239019 52.46559627217389 7.06020836771321 -23.271103117771975 18.969137693675858 -9.91711449785979;
 16.127438562063933 7.06020836771321 50.33769551827648 3.7333899937863997 -6.325975136755001 12.878627213103096;
 -13.343643753797938 -23.271103117771975 3.7333899937863997 39.42319559577969 -2.0943180058706288 9.380291067226182;
 28.189042396975776 18.969137693675858 -6.325975136755001 -2.0943180058706288 74.62886540445582 -4.384778881475672;
 -13.053565290872937 -9.91711449785979 12.878627213103096 9.380291067226182 -4.384778881475672 41.75866065454208;
]
solve(n, X, P0, Q0, M0, true)
## Multiplier-1
P1 = 44519*X[1]^2//1000 - 11637*X[1]*X[2]//1000 - 27071*X[1]//1000 + 25533*X[2]^2//500 + 39727*X[2]//1000 + 31627//500
M1 = [1, X[2], X[1]]
Q1 = [63.25443748506265 19.88941103201886 -13.509723063438985;
 19.88941103201886 51.08230420580756 -5.832400632308046;
 -13.509723063438985 -5.832400632308046 44.50953330380472;
]
solve(n, X, P1, Q1, M1, true)
# unsafe
## Total Decomposition
P2 = 8281*X[1]^4//250 - 29683*X[1]^3*X[2]//10000 - 24793*X[1]^3//1000 + 7479*X[1]^2*X[2]^2//125 + 5096*X[1]^2*X[2]//125 + 69941*X[1]^2//1000 - 75393*X[1]*X[2]^3//10000 - 5499*X[1]*X[2]^2//250 + 53771*X[1]*X[2]//1000 - 14041*X[1]//500 + 9739*X[2]^4//250 + 27751*X[2]^3//500 + 66403*X[2]^2//1000 + 3663*X[2]//250 + 1489//100
M2 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2]
Q2 = [14.889910014161323 7.32719200395884 -14.0376540807002 0.643821196603606 1.621596555024747 1.9813744632556372;
 7.32719200395884 65.12244297697809 25.252263019451192 27.760038359384396 -13.12514830837989 10.507427940649515;
 -14.0376540807002 25.252263019451192 65.96130983053435 2.1184018510330938 9.876243957449278 -12.386696512015247;
 0.643821196603606 27.760038359384396 2.1184018510330938 38.95307805864915 -3.778296111950348 6.293454790744504;
 1.621596555024747 -13.12514830837989 9.876243957449278 -3.778296111950348 47.26250980200498 -1.4878352363383733;
 1.9813744632556372 10.507427940649515 -12.386696512015247 6.293454790744504 -1.4878352363383733 33.103932613471784;
]
solve(n, X, P2, Q2, M2, true)
## Multiplier-1
P3 = 7587*X[1]^2//250 - 1819*X[1]*X[2]//20000 + 6493*X[1]//250 + 3412*X[2]^2//125 - 17941*X[2]//1000 + 10431//200
M3 = [1, X[2], X[1]]
Q3 = [52.1554486865983 -8.965101976197838 12.994028725742382;
 -8.965101976197838 27.29396944858562 -0.04021348550163854;
 12.994028725742382 -0.04021348550163854 30.353059964200792;
]
solve(n, X, P3, Q3, M3, true)
# Lie
## Multiplier-2
P4 = 478*X[1]^4//125 - 7229*X[1]^3*X[2]//1000 - 4199*X[1]^3//1000 + 9807*X[1]^2*X[2]^2//1000 + 8261*X[1]^2*X[2]//1000 + 219*X[1]^2//125 + 59*X[1]*X[2]^3//50 + 971*X[1]*X[2]^2//2500 - 1411*X[1]*X[2]//200 - 357*X[1]//100 + 617*X[2]^4//250 - 1833*X[2]^3//1000 - 3539*X[2]^2//1000 + 9711*X[2]//10000 + 3349//1000
M4 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2]
Q4 = [3.348955107133474 0.48461456903702416 -1.7846778006833879 -2.337752958048619 -2.2376231306734944 -0.6401816791411263;
 0.48461456903702416 1.1366051774850825 -1.2901628737806197 -0.916595236127818 -1.1829811109263162 0.89077248783729;
 -1.7846778006833879 -1.2901628737806197 3.0316706845529273 1.3781337286991575 3.241534107917851 -2.100133325305317;
 -2.337752958048619 -0.916595236127818 1.3781337286991575 2.4681717561246703 0.5904376022292996 0.8402508863894796;
 -2.2376231306734944 -1.1829811109263162 3.241534107917851 0.5904376022292996 8.126896200031702 -3.6148873329673172;
 -0.6401816791411263 0.89077248783729 -2.100133325305317 0.8402508863894796 -3.6148873329673172 3.824366657139208;
]
solve(n, X, P4, Q4, M4, true)
## Total Decomposition
P5 = 300946719944477*X[1]^8//500000000000000 - 24664425291121*X[1]^7*X[2]//31250000000000 - 62251675128937*X[1]^7//50000000000000 + 29837989807129*X[1]^6*X[2]^2//20000000000000 + 317447304725647*X[1]^6*X[2]//100000000000000 + 710135722160339*X[1]^6//100000000000000 - 157137703895569*X[1]^5*X[2]^3//50000000000000 - 11183385848999*X[1]^5*X[2]^2//10000000000000 - 160466480255127*X[1]^5*X[2]//20000000000000 - 899*X[1]^5//200 + 110195231437683*X[1]^4*X[2]^4//25000000000000 + 296566867828369*X[1]^4*X[2]^3//50000000000000 + 207006025314331*X[1]^4*X[2]^2//25000000000000 + 2369*X[1]^4*X[2]//500 + 1957*X[1]^4//1000 - 180246841907501*X[1]^3*X[2]^5//50000000000000 - 685898971557617*X[1]^3*X[2]^4//100000000000000 - 54461*X[1]^3*X[2]^3//8192 - 366*X[1]^3*X[2]^2//25 - 293*X[1]^3*X[2]//20 - 25*X[1]^3//4 + 511593174934387*X[1]^2*X[2]^6//100000000000000 + 27666733264923*X[1]^2*X[2]^5//2500000000000 + 14068265914917*X[1]^2*X[2]^4//1000000000000 + 1083*X[1]^2*X[2]^3//100 + 1749*X[1]^2*X[2]^2//100 + 277*X[1]^2*X[2]//20 + 626*X[1]^2//125 - 137321949005127*X[1]*X[2]^7//50000000000000 - 55410349369049*X[1]*X[2]^6//25000000000000 + 1377*X[1]*X[2]^5//1000 - 199*X[1]*X[2]^4//50 - 8171*X[1]*X[2]^3//1000 - 458*X[1]*X[2]^2//125 - 3943*X[1]*X[2]//500 - 1751*X[1]//500 + 20549653172493*X[2]^8//20000000000000 + 403*X[2]^7//250 - 314*X[2]^6//625 - 1431*X[2]^5//1000 - 4307*X[2]^4//10000 - 2079*X[2]^3//1000 + 747*X[2]^2//500 + 261*X[2]//200 + 3183//1000
M5 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2, X[2]^3, X[1]*X[2]^2, X[1]^2*X[2], X[1]^3, X[2]^4, X[1]*X[2]^3, X[1]^2*X[2]^2, X[1]^3*X[2], X[1]^4]
Q5 = [3.1833561979679064 0.6526058001930102 -1.7508137350988926 -1.5793149849814028 -2.044886681028474 -0.35030020238508286 -0.5128251090547256 -1.2056487899294541 1.517907563755955 -0.8452507851812228 0.16236593112354283 -1.0127082309414106 0.8821274311842332 -0.3007733275264043 0.11856922857165338;
 0.6526058001930102 4.651943713400472 -1.9018304085001396 -0.5276594124911825 -1.775512174104882 1.951170005520043 -2.0199802880043674 -3.2125681440272533 0.9650044167679688 -1.6188527465383677 -0.17205109579934208 -1.7550268367967303 0.4667954972931167 -0.6938677379231639 0.3121770766373375;
 -1.7508137350988926 -1.9018304085001396 5.709476891630792 1.1510084073371656 3.45685380896075 -2.2784074061212567 -0.14120909631841427 3.028873569377462 -2.366277208841464 -1.6081133373130927 -0.6796038158356295 2.1973422952443715 -1.5435553890950016 0.23910394262233342 -0.31415882759816327;
 -1.5793149849814028 -0.5276594124911825 1.1510084073371656 3.2864925405234677 0.2822429961236751 -0.0692233722574216 -0.5458216531307005 0.431784648791588 -1.13171961580537 0.6087512900527882 -1.2482629234670537 1.3601660190669895 -0.28481948333596063 0.1823400829978368 -0.1381464398685692;
 -2.044886681028474 -1.775512174104882 3.45685380896075 0.2822429961236751 7.879921526037149 -3.0403496338021534 0.012698707861953014 3.868682923190193 -2.8532318803378587 0.32880376085304025 -0.5127082033021816 2.0939335291808963 -1.0672660170861534 0.3514625990274026 -0.5310229250686709;
 -0.35030020238508286 1.951170005520043 -2.2784074061212567 -0.0692233722574216 -3.0403496338021534 4.935185742580686 0.013813763750746598 -2.839068584407965 1.4865969188851484 -1.931029211148678 0.48550343950029484 -1.6340193180554377 0.4296091015428149 -1.109485136804938 0.49134007532997476;
 -0.5128251090547256 -2.0199802880043674 -0.14120909631841427 -0.5458216531307005 0.012698707861953014 0.013813763750746598 1.9969337215379677 -0.1578374412694783 0.005554254988639481 1.3310054971466605 0.8058232188808147 -0.35772620092804996 0.2754350051995876 0.23094828812338236 -0.05398562694626945;
 -1.2056487899294541 -3.2125681440272533 3.028873569377462 0.431784648791588 3.868682923190193 -2.839068584407965 -0.1578374412694783 9.4681602208505 -2.132639707646103 0.3320882483848152 -0.7446941062408375 4.5708497335354 -1.9091952300159603 0.8914578568720692 -0.45284196128105636;
 1.517907563755955 0.9650044167679688 -2.366277208841464 -1.13171961580537 -2.8532318803378587 1.4865969188851484 0.005554254988639481 -2.132639707646103 6.327712145503259 -2.3684221723326595 0.6856218532857117 -1.8342868192447546 1.4955539909671156 -0.553635542463412 0.426303219547528;
 -0.8452507851812228 -1.6188527465383677 -1.6081133373130927 0.6087512900527882 0.32880376085304025 -1.931029211148678 1.3310054971466605 0.3320882483848152 -2.3684221723326595 6.115226265251867 0.08596540097946997 0.6330822579329646 0.4494100041827922 1.1595338202832937 -0.6229920928409765;
 0.16236593112354283 -0.17205109579934208 -0.6796038158356295 -1.2482629234670537 -0.5127082033021816 0.48550343950029484 0.8058232188808147 -0.7446941062408375 0.6856218532857117 0.08596540097946997 1.0253845189991289 -1.3697521284703835 0.12560948999132404 0.13264704603253727 0.18311392406702712;
 -1.0127082309414106 -1.7550268367967303 2.1973422952443715 1.3601660190669895 2.0939335291808963 -1.6340193180554377 -0.35772620092804996 4.5708497335354 -1.8342868192447546 0.6330822579329646 -1.3697521284703835 4.860943534498217 -1.9334955170561294 0.011391109010569484 -0.34560055945184087;
 0.8821274311842332 0.4667954972931167 -1.5435553890950016 -0.28481948333596063 -1.0672660170861534 0.4296091015428149 0.2754350051995876 -1.9091952300159603 1.4955539909671156 0.4494100041827922 0.12560948999132404 -1.9334955170561294 4.020014197948409 -1.2252814549334063 -0.32598959672639943;
 -0.3007733275264043 -0.6938677379231639 0.23910394262233342 0.1823400829978368 0.3514625990274026 -1.109485136804938 0.23094828812338236 0.8914578568720692 -0.553635542463412 1.1595338202832937 0.13264704603253727 0.011391109010569484 -1.2252814549334063 2.1425286983850818 -0.3943162196294828;
 0.11856922857165338 0.3121770766373375 -0.31415882759816327 -0.1381464398685692 -0.5310229250686709 0.49134007532997476 -0.05398562694626945 -0.45284196128105636 0.426303219547528 -0.6229920928409765 0.18311392406702712 -0.34560055945184087 -0.32598959672639943 -0.3943162196294828 0.6022518201891683;
]
solve(n, X, P5, Q5, M5, true)
## Multiplier-1
P6 = 299*X[1]^4//100 - 5841*X[1]^3*X[2]//1000 - 3879*X[1]^3//1000 + 9441*X[1]^2*X[2]^2//1000 + 7161*X[1]^2*X[2]//1000 + 2201*X[1]^2//1000 + 1309*X[1]*X[2]^3//5000 + 5973*X[1]*X[2]^2//10000 - 1691*X[1]*X[2]//250 - 29*X[1]//10 + 8307*X[2]^4//10000 + 1503*X[2]^3//1000000 - 1653*X[2]^2//1000 - 7073*X[2]//10000 + 29//10
M6 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2]
Q6 = [2.9013639515541065 -0.35258277857141906 -1.4502537048443525 -1.0691848794212842 -2.9480554073647514 -0.1452562675177013;
 -0.35258277857141906 0.4869402247504604 -0.4359415039545594 0.0002497905940219366 -0.14279374562209382 0.6271056813057706;
 -1.4502537048443525 -0.4359415039545594 2.4885162365556703 0.43923727433857107 2.9530657326932532 -1.93799030155431;
 -1.0691848794212842 0.0002497905940219366 0.43923727433857107 0.8303331159414751 0.12992977559688149 0.51789354397035;
 -2.9480554073647514 -0.14279374562209382 2.9530657326932532 0.12992977559688149 8.407596235487421 -2.920083177700133;
 -0.1452562675177013 0.6271056813057706 -1.93799030155431 0.51789354397035 -2.920083177700133 2.9901218431910443;
]
solve(n, X, P6, Q6, M6, true)

@show SOS_time Newton_time