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


#H7
# third
## Total Decomposition
P0 = 1103*X[1]^4//12500 - 251*X[1]^3*X[2]//625 + 391*X[1]^3//1250 + 529*X[1]^2*X[2]^2//1000 - 249*X[1]^2*X[2]//200 + 2367*X[1]^2//2500 + 19*X[1]*X[2]^3//80 + 7993*X[1]*X[2]^2//10000 - 1187*X[1]*X[2]//1000 - 319*X[1]//200 + 6457*X[2]^4//100000 + 3171*X[2]^3//5000 + 179*X[2]^2//100 - 383*X[2]//625 + 641//500
M0 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2]
Q0 = [1.2824197749315538 -0.3062146817883028 -0.7975802483037402 -0.0629787575324833 0.04815039518275669 -0.051443358892777624;
 -0.3062146817883028 1.9147117663195938 -0.641165050038347 0.31647879592603206 0.5107638580648131 -0.21440554234554682;
 -0.7975802483037402 -0.641165050038347 1.049009593756288 -0.11129883708688484 -0.40730735205257385 0.15660208888254162;
 -0.0629787575324833 0.31647879592603206 -0.11129883708688484 0.06445788921718272 0.11863110234098226 -0.044054205762102006;
 0.04815039518275669 0.5107638580648131 -0.40730735205257385 0.11863110234098226 0.6174582683682439 -0.20091637779369012;
 -0.051443358892777624 -0.21440554234554682 0.15660208888254162 -0.044054205762102006 -0.20091637779369012 0.08787367888539092;
]
solve(n, X, P0, Q0, M0, true)
## Multiplier-1
P1 = 1401*X[1]^2//500 - 19*X[1]*X[2]//10 - 1289*X[1]//250 + 561*X[2]^2//500 + 509*X[2]//500 + 1347//500
M1 = [1, X[2], X[1]]
Q1 = [2.6941437765951095 0.5098500911316238 -2.5792736173509243;
 0.5098500911316238 1.1225232229850903 -0.9508428967775737;
 -2.5792736173509243 -0.9508428967775737 2.803578750175814;
]
solve(n, X, P1, Q1, M1, true)
# second
## Total Decomposition
P2 = 153*X[1]^4//100 - 1481*X[1]^3*X[2]//5000 - 2959*X[1]^3//10000 + 4917*X[1]^2*X[2]^2//5000 + 1553*X[1]^2*X[2]//10000 + 9*X[1]^2//5 - 1553*X[1]*X[2]^3//1000 - 147*X[1]*X[2]^2//500 - 11*X[1]*X[2]//400 - 197*X[1]//500 + 8117*X[2]^4//10000 - 18*X[2]^3//125 + 159*X[2]^2//250 + 201*X[2]//5000 + 347//500
M2 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2]
Q2 = [0.6944783034266785 0.020477922791178736 -0.1974746900101466 -0.13111807405008605 0.14661462363835992 0.10665926037609548;
 0.020477922791178736 0.89902399553398 -0.16079600886674264 -0.07255461930118984 -0.0886883183993241 0.02177145637463683;
 -0.1974746900101466 -0.16079600886674264 1.5864711928024033 -0.057080383659227435 0.055923696385671934 -0.148253540703999;
 -0.13111807405008605 -0.07255461930118984 -0.057080383659227435 0.811196291810271 -0.7758816672979638 -0.30196342109955393;
 0.14661462363835992 -0.0886883183993241 0.055923696385671934 -0.7758816672979638 1.5872187319675097 -0.148607649574895;
 0.10665926037609548 0.02177145637463683 -0.148253540703999 -0.30196342109955393 -0.148607649574895 1.5300816301116675;
]
solve(n, X, P2, Q2, M2, true)
## Multiplier-1
P3 = 1747*X[1]^2//1000 + 151*X[1]*X[2]//250 + 6591*X[1]//10000 + 551*X[2]^2//250 + 428*X[2]//625 + 107//50
M3 = [1, X[2], X[1]]
Q3 = [2.139510177656067 0.3426280599518026 0.3287744136446899;
 0.3426280599518026 2.203694055963389 0.3023068026596257;
 0.3287744136446899 0.3023068026596257 1.74629718620365;
]
solve(n, X, P3, Q3, M3, true)
# seventh
## Total Decomposition
P4 = 657*X[1]^4//1000 + 1313*X[1]^2*X[2]^2//1000 + 263*X[1]^2//500 + 657*X[2]^4//1000 + 263*X[2]^2//500 + 217//250
M4 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2]
Q4 = [0.8676006530858223 -8.89532400637798e-18 1.7502628104577824e-18 -0.11410083916571143 -3.759481026525056e-17 -0.11410083916571138;
 -8.89532400637798e-18 0.7551248700091886 4.023528997440796e-16 2.1934788972757846e-17 -8.329255804209893e-17 -1.517827443948744e-16;
 1.7502628104577824e-18 4.023528997440796e-16 0.7551248700091894 1.7675124434921695e-16 3.462317982954823e-17 -1.9459196372007226e-16;
 -0.11410083916571143 2.1934788972757846e-17 1.7675124434921695e-16 0.6567310247802182 -2.656492410043168e-19 0.10577727233888114;
 -3.759481026525056e-17 -8.329255804209893e-17 3.462317982954823e-17 -2.656492410043168e-19 1.1019075048501097 -8.730234304804646e-19;
 -0.11410083916571138 -1.517827443948744e-16 -1.9459196372007226e-16 0.10577727233888114 -8.730234304804646e-19 0.6567310247802184;
]
solve(n, X, P4, Q4, M4, true)
## Multiplier-1
P5 = 657*X[1]^2//1000 + 657*X[2]^2//1000 + 691//1000
M5 = [1, X[2], X[1]]
Q5 = [0.6911059478646795 2.1262752760677008e-18 -7.00105124183204e-18;
 2.1262752760677008e-18 0.6567310247802182 -3.628131357202994e-19;
 -7.00105124183204e-18 -3.628131357202994e-19 0.6567310247802179;
]
solve(n, X, P5, Q5, M5, true)
# eighth
## Total Decomposition
P6 = 783*X[1]^4//2500 - 6563*X[1]^3*X[2]//1000000 - 1089*X[1]^3//1000 + 388*X[1]^2*X[2]^2//625 - 2487*X[1]^2*X[2]//2500 + 581*X[1]^2//200 - 1349*X[1]*X[2]^3//200000 - 523*X[1]*X[2]^2//500 - 329*X[1]*X[2]//200 + 8433*X[1]//10000 + 3087*X[2]^4//10000 - 503*X[2]^3//500 + 617*X[2]^2//200 + 7819*X[2]//10000 + 183//125
M6 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2]
Q6 = [1.4643384563085984 0.39046440491925627 0.4214106146598555 0.005979329215855394 -0.442660450664223 -0.03186055588142249;
 0.39046440491925627 3.0735291830906974 -0.3800504511746531 -0.5045886642552503 -0.5368751262995239 -0.00106884228969064;
 0.4214106146598555 -0.3800504511746531 2.9684961438334274 0.013685634032952169 -0.4989850834777631 -0.5432496193027523;
 0.005979329215855394 -0.5045886642552503 0.013685634032952169 0.3090029942910307 -0.0030743581904994753 -0.04800779586274819;
 -0.442660450664223 -0.5368751262995239 -0.4989850834777631 -0.0030743581904994753 0.7179029274446532 -0.003074358190499012;
 -0.03186055588142249 -0.00106884228969064 -0.5432496193027523 -0.04800779586274819 -0.003074358190499012 0.31288434142811844;
]
solve(n, X, P6, Q6, M6, true)
## Multiplier-1
P7 = 3131*X[1]^2//10000 - 6159*X[1]*X[2]//1000000 + 331*X[1]//2000 + 619*X[2]^2//2000 + 2277*X[2]//10000 + 207//200
M7 = [1, X[2], X[1]]
Q7 = [1.0346663519771933 0.11341732432680546 0.08251906355349191;
 0.11341732432680546 0.30900299429102945 -0.0030743581905010595;
 0.08251906355349191 -0.0030743581905010595 0.3128843414281213;
]
solve(n, X, P7, Q7, M7, true)
# first
## Total Decomposition
P8 = 411*X[1]^4//1250 + 427*X[1]^3*X[2]//10000 + 1169*X[1]^3//1000 + 6839*X[1]^2*X[2]^2//10000 - 1169*X[1]^2*X[2]//1250 + 423*X[1]^2//125 + 138*X[1]*X[2]^3//3125 + 138*X[1]*X[2]^2//125 + 987*X[1]*X[2]//2000 - 337*X[1]//250 + 3571*X[2]^4//10000 - 1223*X[2]^3//1000 + 2731*X[2]^2//1000 + 939*X[2]//2000 + 2697//1000
M8 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2]
Q8 = [2.6972648839265423 0.23585067120403236 -0.6736880267756274 -0.013806786024998966 0.4418087437109648 0.032470975060849595;
 0.23585067120403236 2.759061488014938 -0.19537840196334558 -0.6110120918884911 0.40804559291413695 -0.027883510432084205;
 -0.6736880267756274 -0.19537840196334558 3.3193555343492758 0.14467452693655447 -0.4400075530600429 0.5837566514524883;
 -0.013806786024998966 -0.6110120918884911 0.14467452693655447 0.3562892129037466 0.021769694999769804 -0.02510588370486908;
 0.4418087437109648 0.40804559291413695 -0.4400075530600429 0.021769694999769804 0.7347690690185763 0.021769694999770446;
 0.032470975060849595 -0.027883510432084205 0.5837566514524883 -0.02510588370486908 0.021769694999770446 0.3282680887050971;
]
solve(n, X, P8, Q8, M8, true)
## Multiplier-1
P9 = 41*X[1]^2//125 + 439*X[1]*X[2]//10000 - 73*X[1]//500 + 357*X[2]^2//1000 + 203*X[2]//1000 + 63//200
M9 = [1, X[2], X[1]]
Q9 = [0.31500935554927245 0.10156633391897843 -0.07277952595770049;
 0.10156633391897843 0.3562892129037453 0.021769694999772733;
 -0.07277952595770049 0.021769694999772733 0.3282680887050968;
]
solve(n, X, P9, Q9, M9, true)
# fifth
## Total Decomposition
P10 = 49*X[1]^4//25 + 509*X[1]^3*X[2]//12500 - 5341*X[1]^3//10000 + 3857*X[1]^2*X[2]^2//1000 - 1633*X[1]^2*X[2]//5000 + 2117*X[1]^2//1000 + 3097*X[1]*X[2]^3//50000 + 103*X[1]*X[2]^2//125 + 1581*X[1]*X[2]//5000 - 227*X[1]//625 + 207*X[2]^4//100 + 201*X[2]^3//200 + 3959*X[2]^2//1000 + 1169*X[2]//1000 + 3077//1000
M10 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2]
Q10 = [3.0767906150162387 0.5834380668724043 -0.18302225848718076 0.29612039860468964 0.16094660265913174 -0.16042904058706675;
 0.5834380668724043 3.3680102770057125 -0.0030535028546297377 0.5012013452154515 0.3697930968533862 -0.024010207332751374;
 -0.18302225848718076 -0.0030535028546297377 2.436793120027532 0.0418634600700921 -0.14073388875137455 -0.26809970336706457;
 0.29612039860468964 0.5012013452154515 0.0418634600700921 2.069982882255826 0.03015799542986956 0.24886230722745997;
 0.16094660265913174 0.3697930968533862 -0.14073388875137455 0.03015799542986956 3.360781106480735 0.020856159133598704;
 -0.16042904058706675 -0.024010207332751374 -0.26809970336706457 0.24886230722745997 0.020856159133598704 1.9602603441434001;
]
solve(n, X, P10, Q10, M10, true)
## Multiplier-1
P11 = 989*X[1]^2//500 + 3013*X[1]*X[2]//100000 + 313*X[1]//2500 + 391*X[2]^2//200 - 151*X[2]//1000 + 1841//1000
M11 = [1, X[2], X[1]]
Q11 = [1.8412368080485486 -0.07545700349120754 0.06285269340574963;
 -0.07545700349120754 1.955157975829853 0.014897377918003192;
 0.06285269340574963 0.014897377918003192 1.9778499157848277;
]
solve(n, X, P11, Q11, M11, true)
# fourth
## Total Decomposition
P12 = 2943*X[1]^4//1000 + 1143*X[1]^3*X[2]//10000 - 603*X[1]^3//1250 + 471*X[1]^2*X[2]^2//100 - 4321*X[1]^2*X[2]//10000 + 3169*X[1]^2//1000 - 4187*X[1]*X[2]^3//100000 - 597*X[1]*X[2]^2//2000 - 457*X[1]*X[2]//250 + 1577*X[1]//1000 + 107*X[2]^4//40 + 8207*X[2]^3//100000 + 63*X[2]^2//40 - 57*X[2]//20 + 2629//500
M12 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2]
Q12 = [5.2581207558738035 -1.4272893967329598 0.7883209414032136 -0.8006539715137404 -0.45988568089524845 -0.15025137862546362;
 -1.4272893967329598 3.1742428565439176 -0.45509208562494674 0.04312086122033261 0.05002293837130016 -0.016933727513142732;
 0.7883209414032136 -0.45509208562494674 3.4726831071663 -0.19896919961123022 -0.2007872524589969 -0.24157029941653546;
 -0.8006539715137404 0.04312086122033261 -0.19896919961123022 2.6756879709149395 -0.02103332166265707 0.3162776038722883;
 -0.45988568089524845 0.05002293837130016 -0.2007872524589969 -0.02103332166265707 4.075065934213657 0.057749844049254984;
 -0.15025137862546362 -0.016933727513142732 -0.24157029941653546 0.3162776038722883 0.057749844049254984 2.9429056140143697;
]
solve(n, X, P12, Q12, M12, true)
## Multiplier-1
P13 = 307*X[1]^2//125 - 2031*X[1]*X[2]//50000 + 1787*X[1]//5000 + 2563*X[2]^2//1000 - 7493*X[2]//100000 + 1441//500
M13 = [1, X[2], X[1]]
Q13 = [2.882195378770574 -0.036294038303429144 0.17992047425329255;
 -0.036294038303429144 2.5635334528147493 -0.02030008766311253;
 0.17992047425329255 -0.02030008766311253 2.4567489773897098;
]
solve(n, X, P13, Q13, M13, true)
# sixth
## Total Decomposition
P14 = 547*X[1]^4//1000 + 109*X[1]^2*X[2]^2//100 + 157*X[1]^2//500 + 547*X[2]^4//1000 + 157*X[2]^2//500 + 171//250
M14 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2]
Q14 = [0.6840732083612268 2.92960783994876e-17 -3.280041704252147e-18 -0.13127639312129463 1.860998937476463e-17 -0.13127639312129477;
 2.92960783994876e-17 0.5766985297300306 -1.1579687064153913e-16 -2.0085267652949402e-16 3.2876278973593615e-17 -1.879865925846067e-16;
 -3.280041704252147e-18 -1.1579687064153913e-16 0.5766985297300308 1.174706700910314e-16 2.4547118847052017e-17 -1.255746434722236e-16;
 -0.13127639312129463 -2.0085267652949402e-16 1.174706700910314e-16 0.5472206475805365 2.4189938806079364e-18 0.09708942162376542;
 1.860998937476463e-17 3.2876278973593615e-17 2.4547118847052017e-17 2.4189938806079364e-18 0.9002624517905241 2.4664048970916513e-18;
 -0.13127639312129477 -1.879865925846067e-16 -1.255746434722236e-16 0.09708942162376542 2.4664048970916513e-18 0.5472206475805368;
]
solve(n, X, P14, Q14, M14, true)
## Multiplier-1
P15 = 547*X[1]^2//1000 + 547*X[2]^2//1000 + 311//500
M15 = [1, X[2], X[1]]
Q15 = [0.6219573576822949 -4.646663159112889e-17 5.831185251930462e-18;
 -4.646663159112889e-17 0.5472206475805368 -6.926231726491798e-18;
 5.831185251930462e-18 -6.926231726491798e-18 0.5472206475805367;
]
solve(n, X, P15, Q15, M15, true)

@show SOS_time Newton_time