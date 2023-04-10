% CPSC 312 Calculus and Algebra in Prolog
% Copyright D. Poole 2023. Released under GPL https://www.gnu.org/licenses/gpl-3.0.en.html
  
% An expression can include algebaic variables, which are Prolog constants

% use if cant run pce
%
%assertz(file_search_path(library,pce('prolog/lib'))).
%assertz(file_search_path(pce,swi(xpce))).
:- use_module(library(pce)).

ask_name(Name) :-
        new(D, dialog('Register')),
        send(D, scrollbars, both),
        send(D, append(new(N1, text_item(equation)))),
        send(D, append(new(N2, text_item(answer)))),
        send(D, append(new(N3, text_item(respect)))),
        send(D, append(new(S, new(S, menu(function))))),
        send(S, append, eval),
        send(S, append, simplify),
        send(S, append, deriv),
        send(S, append, integ),
        send(D, append(button(enter, and(message(@prolog, output,
                                          N1?selection,N2?selection,N3?selection, S?selection))))),
        send(D, append(button(cancel, message(D, return, @nil)))),

        send(D, layout),
        send(D, size, size(600,600)),

        get(D, confirm, Rval),
        Rval \== @nil, % canceled
        Name = Rval,
        free(D).

output(E,A,R, integ) :- integ(E, R, A).
output(E,A,R, eval) :- foo(E, R, A).

integ(0,0,0) :- format('[Step] We resolve the value').
%integ(A,C,V) :- format('[Step] We resolve the value').

foo(A,C,V) :- format('[Step] We resolve the value').

%eval(Exp, Env, V) is true if expression Exp evaluates to V given environment Env
% An environment is a list of val(Var,Val) indicating that variable Var has value Val
% eval supports addition, subtraction, division, multiplication operations of constant, polynomial, sine/cosine, natural logarithmic, and exponential functions.
eval(X,Env,V) :-
    member(val(X,V),Env),
    format('[Step] We resolve the value of ~w = ~f~n', [X,V]). 
eval(N,_,N) :-
    number(N). %No steps needed, a number is a number.
eval((A+B),Env,V) :-
    eval(A,Env,VA),
    eval(B,Env,VB),
    V is VA+VB,
    format('[Step] We take the sum of ~w = ~f and ~w = ~f to get ~f~n', [A,VA,B,VB,V]). 
eval((A*B),Env,V) :-
    eval(A,Env,VA),
    eval(B,Env,VB),
    V is VA*VB,
    format('[Step] We take the product of ~w = ~f and ~w = ~f to get ~f~n', [A,VA,B,VB,V]). 
eval((A/B),Env,V) :-
    eval(A,Env,VA),
    eval(B,Env,VB),
    V is VA/VB,
    format('[Step] We divide ~w = ~f by ~w = ~f to get ~f~n', [A,VA,B,VB,V]). 
eval((A-B),Env,V) :-
    eval(A,Env,VA),
    eval(B,Env,VB),
    V is VA-VB,
    format('[Step] We subtract ~w = ~f from ~w = ~f to get ~f~n', [B,VB,A,VA,V]). 
eval(-A,Env,V) :-
    eval(A,Env,VA),
    V is -VA,
    format('[Step] We negate ~w = ~f to get ~f~n', [A,VA,V]). 

eval((A^B),Env,V) :-
    eval(A,Env,VA),
    eval(B,Env,VB),
    V is VA^VB,
    format('[Step] We take ~w = ~f to the power of ~w = ~f to get ~f~n', [A,VA,B,VB,V]). 
eval(log(A),Env,V) :-
    eval(A,Env,VA),
    V is log(VA),
    format('[Step] We take the natural log of ~w = ~f in log(~w) to get ~f~n', [A,VA,A,V]).
eval(exp(A),Env,V) :-
    eval(A,Env,VA),
    V is exp(VA),
   format('[Step] We get the natural exponent of ~w (e^~w) to get ~f~n', [A,VA,V]).
eval(sigmoid(A),Env,V) :-
    eval(A,Env,VA),
    V is 1/(1+exp(-VA)),
    format('[Step] We get the sigmoid of ~w = ~f in exp(~w)/(1 + exp(~w)) to get ~f~n', [A,VA,A,A,V]).
eval(sin(A),Env,V) :-
    eval(A,Env,VA),
    V is sin(VA),
    format('[Step] We get the sine of ~w = sin(~f) = ~f~n', [A,VA,V]).
eval(cos(A),Env,V) :-
    eval(A,Env,VA),
    V is cos(VA),
    format('[Step] We get the cosine of ~w = cos(~f) = ~f~n', [A,VA,V]).
eval(exp(A),Env,V) :-
    eval(A,Env,VA),
    V is exp(VA),
    format('[Step] We get the exponent of ~w (e^~w)= exp(~f) = ~f~n', [A,VA,VA,V]).

% For demo:
% Eval queries with steps:
% Basic exmaple: eval(a^2 + b^2 ,[val(a,3),val(b,4)],V). 
% Sine, cosine square sum identity: eval(sin(x)^2 + cos(x)^2,[val(x,12)],V).  
% log - exp property: eval(log(exp(x)),[val(x,120)],V).
% Sigmoid is e^x / (1+e^x), neat example: eval(sigmoid(x),[val(x,0)],V). 


% Derivative demo queries:
% sine derivate: deriv(sin(3.14*y),y,V),eval(V,[val(y,1)],X).
% polynomial derivative: deriv(y^4 + y^2 + y + 21,y,V),eval(V,[val(y,17)],X).    
% exponential derivative: deriv(exp(y),y,V),eval(V,[val(y,1)],X).  
% Multivariable Polynomial derivative: deriv(y^4+x,y,V),eval(V,[val(y,17),val(x,1)],X). 

% Integral demo queries:
% Polynomial integral: integ(y^4,y,V),eval(V,[val(y,17)],X). 
% Multivariable Polynomial integral: integ(y^4+x,y,V),eval(V,[val(y,17),val(x,1)],X).
% Integral of exponent function: integ(exp(y),y,V),eval(V,[val(y,1)],X).  
% Logarithmic integral: integ(log(y),y,V),eval(V,[val(y,17)],X).

%  Differentiation
% deriv(E,X,DE) is true if DE is the derivative of E with respect to X
deriv(X,X,1) :-
    format('[Step] Derivative d/d~w ~w = ~f~n', [X,X,1]). 
deriv(C,X,0) :- 
    atomic(C),
    dif(C,X),
    format('[Step] Derivative of a constant: d/d~w ~w = ~f~n', [X,C,0]).
deriv(A+B,X,DA+DB) :-
    deriv(A,X,DA),
    deriv(B,X,DB),
    format('[Step] Derivative of sums is sum of derivatives: d/d~w (~w + ~w) = d/d~w ~w + d/d~w ~w = ~w~n', [X,A,B,X,A,X,B,DA+DB]).
deriv(A-B,X,DA-DB) :-
    deriv(A,X,DA),
    deriv(B,X,DB),
    format('[Step] Derivative of sums is sum of derivatives: d/d~w (~w - ~w) = d/d~w ~w - d/d~w ~w = ~w~n', [X,A,B,X,A,X,B,DA-DB]).
deriv(A*B,X,A*DB+B*DA) :-
    deriv(A,X,DA),
    deriv(B,X,DB),
    format('[Step] Derivative of products: d/d~w (~w*~w) = ~w d/d~w ~w + ~w d/d~w ~w = ~w~n', [X,A,B,A,X,B,B,X,A,A*DB+B*DA]).
deriv(A/B,X,(B*DA-A*DB)/(B*B)) :-
    deriv(A,X,DA),
    deriv(B,X,DB),
    format('[Step] Derivative of quotient: d/d~w (~w/~w) = (~w d/dx ~w - ~w d/dx ~w)/(~w*~w) = ~w~n', [X,A,B,B,A,A,B,B,B,(B*DA-A*DB)/(B*B)]).

deriv(-A,X,-DA) :-
    deriv(A,X,DA),
    format('[Step] Derivative of d/d~w ~w = ~w~n', [X,-A,-DA]). 
deriv(A^B,X,B*(A^C)*DA) :-  % only works when B does not involve X
    deriv(A,X,DA),
    simplify((B-1),C),
    format('[Step] Derivative Chain Rule: d/d~w ~w = ~w ~n', [X,A^B,B*(A^C)*DA]). 
deriv(sin(E),X,cos(E)*DE) :-
    deriv(E,X,DE),
    format('[Step] Derivative of sine is cosine: d/d~w ~w = ~w~n', [X,sin(E),cos(E)*DE]). 
deriv(cos(E),X,-sin(E)*DE) :-
    deriv(E,X,DE),
    format('[Step] Derivative of cosine is negative sine: d/d~w ~w = ~w~n', [X,cos(E),-sin(E)*DE]). 
deriv(exp(E),X,exp(E)*DE) :-
    deriv(E,X,DE),
    format('[Step] Derivative of exponent: d/d~w ~w = exp(~w) ~n', [X,exp(E),exp(E)*DE]). 
deriv(log(E),X,DE/E) :-
  deriv(E,X,DE),
  format('[Step] Derivative of log: d/d~w ~w = ~w~n', [X,log(E),DE/E]). 
deriv(sigmoid(E),X,sigmoid(E)*(1-sigmoid(E))*DE) :-
    deriv(E,X,DE),
    format('[Step] Derivative of sigmoid: d/d~w ~w = ~w~n', [X,sigmoid(E),sigmoid(E)*(1-sigmoid(E))*DE]). 


% Some Examples to try:
%?- deriv(x+3*x+6*x*y, x, D).
%?- deriv(7+3*x+6*x*y, x, D).
%?- deriv(x+3*x+6*x*y+ 11*x*x, x, D).
%?- deriv(1/(1+exp(-x)),x,D), simplify(D,E).

% Multi-variate calculus:
%?- deriv(x+3*x+6*x*y+ 11*x*x, x, Dx), deriv(Dx,y,Dxy), simplify(Dxy,E).

% integ(E,X,IE) is true if IE is the integral of E with respect to X


% Basic Integral Rules
integ(0, X, 0) :-
    format('[Step] Integral of ~w w.r.t. ~w = ~w~n', [0,X,0]).
integ(C, X, C*X) :- 
    atomic(C),
    dif(C, X),
    format('[Step] Integral of a constant ~w w.r.t. ~w = ~w~n', [C,X,C*X]).
integ(X, X, X^2/2) :-
    format('[Step] Integral of ~w w.r.t. ~w = ~w~n', [X,X,X^2/2]).
integ(X^N, X, (X^A)/A) :- 
    atomic(N), 
    A is N+1,
    format('[Step] Integral of ~w w.r.t. ~w = ~w~n', [X^N,X,(X^A)/A]).
integ(C/X, X, C*log(abs(X))) :-
    format('[Step] Integral of ~w w.r.t. ~w = ~w~n', [C/X,X,C*log(abs(X))]).
integ(exp(X), X, exp(X)) :-
    format('[Step] Integral of exponent ~w w.r.t. ~w = ~w~n', [exp(X),X,exp(X)]).
integ(C^X, X, (C^X)/log(C)) :- 
    atomic(C),
    dif(C,X),
    format('[Step] Integral of constant to the power of a variable: ~w w.r.t. ~w = ~w~n', [C^X, X, (C^X)/log(C)]).
integ(log(X), X, X*log(X)-X) :-
    format('[Step] Integral of log: ~w w.r.t. ~w = ~w~n', [log(X), X, X*log(X)-X]).

% DEMO
% integ(0, X, 0).
% integ(5, X, 5*X).
% integ(X^4, X, (X^5)/5).

integ(C*A,X,C*IA) :-
    atomic(C),
    integ(A, X, IA),
    format('[Step] Integral of constant multiple: ~w w.r.t. ~w = ~w~n', [C*A,X,C*IA]).
integ(A+B,X,IA+IB) :-
    integ(A,X,IA),
    integ(B,X,IB),
    format('[Step] Integral of sum is sum of integrals: Integral of ~w w.r.t. ~w = ~w~n', [A+B,X,IA+IB]).
integ(A-B,X,IA-IB) :-
    integ(A,X,IA),
    integ(B,X,IB),
    format('[Step] Integral of sum is sum of integrals: Integral of ~w w.r.t. ~w = ~w~n', [A-B,X,IA-IB]).


% Gradient descent with fixed step size, needs a small enough tolerance for it to be accurate.
% VAR is the variable of the function i.e. x
% ALPHA is step size
% TOL is tolerance, For these examples we picked 10^-8
% FPRIME is derivative of F w.r.t VAR
% FPRIMEX is FPRIME evaluated at X
% ITERS is number of iterations
% MIN is the result we want: the minimum value the gradient descent function finds.
gradDescent(F, VAR, ALPHA, TOL, X, ITERS, MIN) :- 
    deriv(F,VAR,FPRIME),
    eval(FPRIME,[val(VAR,X)],FPRIMEX),
    eval(F,[val(VAR,X)],CMIN),
    X1 is X - ALPHA * FPRIMEX,
    format('[Gradient Descent ~d] At ~w = ~f | f(~w) = ~f | f\'(~w) = ~f~n', [ITERS,VAR,X,VAR,CMIN,VAR,FPRIMEX]),
    ((abs(X1-X) < TOL) -> eval(F,[val(VAR,X1)],MIN) ;
    NITERS is ITERS+1,
    gradDescent(F,VAR,ALPHA,TOL,X1, NITERS, MIN)
    ).

% Gradient Descent demo queries:
% gradDescent(x^2, x, 0.01, 0.00000001,10, 0, MIN).
% gradDescent(sin(x), x, 0.01, 0.00000001,10, 0, MIN).
% gradDescent(x^2 - 3*x + 4, 0.01, 0.00000001,10, 0, MIN).
% Complex example: deriv(1/3*x^3, x, K), gradDescent(K, x, 0.01, 0.00000001, 10, 0, MIN).

% DEMO
% integ(8*x, x, I).
% integ(2*x+x, x, I).
% integ(2*x-x, x, I).

%simplify(Exp, Exp2) is true if expression Exp2 is a simplifed form of Exp
simplify(X,X) :-
    atomic(X).
simplify((A+B),V) :-
    simplify(A, VA),
    simplify(B, VB),
    simp_vals(VA+VB, V).
simplify((A*B),V) :-
    simplify(A, VA),
    simplify(B, VB),
    simp_vals(VA*VB, V).
simplify((A/B),V) :-
    simplify(A, VA),
    simplify(B, VB),
    simp_vals(VA/VB, V).
simplify((A-B),V) :-
    simplify(A, VA),
    simplify(B, VB),
    simp_vals(VA-VB, V).
simplify(-E,R) :-
    simplify(E,S),
    simp_vals(-S,R).
simplify(sigmoid(E),sigmoid(S)) :-
    simplify(E,S).
simplify(log(E),log(S)) :-
    simplify(E,S).
simplify(exp(E),exp(S)) :-
    simplify(E,S).
simplify(A^B,S^B) :-
    simplify(A,S).

%simp_vals(Exp, Exp2) is true if expression Exp simplifies to Exp2,
% where the arguments to Exp have already been simplified
% Note last clause is a catch-all.
simp_vals(0+V,V).
simp_vals(V+0,V).
simp_vals(V-0,V).
simp_vals(0-V,- V).
simp_vals(-(0),0).
simp_vals(-(-X),X).
simp_vals(A+B,AB) :-
    number(A),number(B),
    AB is A+B.
simp_vals(A-B,AB) :-
    number(A),number(B),
    AB is A-B.
simp_vals(0*_,0).
simp_vals(_*0,0).
simp_vals(_*(-0),0).
simp_vals(_*(-(0)),0).
simp_vals(V*1,V).
simp_vals(1*V,V).
simp_vals(A*B,AB) :-
    number(A),number(B),
    AB is A*B.
simp_vals(0/_,0).
simp_vals(V/1,V).
simp_vals(A/B,AB) :-
    number(A),number(B),
    AB is A/B.
simp_vals(X,X).

% try:
%?- simplify(y*1+(0*x + x*0),E).
%?- simplify(y*(2*10+3),E).
%?- simplify(1+ (3*1+x*0)+ (6*x*0+y* (6*1+x*0))+ (11*x*1+x* (11*1+x*0)), E).

% Examples from learning (some that I used when building a learning system)
% deriv(-log(sigmoid(w3)*phgr+sigmoid(w4)*(1-phgr)),w3,D), simplify(D,S).
% deriv(-log(sigmoid(w3)*phgr+sigmoid(w4)*(1-phgr)),w4,D), simplify(D,S).
% deriv(-log(sigmoid(w4) + (sigmoid(w3)-sigmoid(w4))*sigmoid(w0+w1*pr+w2*nr)),w0,D),simplify(D,S).
% % deriv(-log(sigmoid(w4) + (sigmoid(w3)-sigmoid(w4))*sigmoid(w0+w1*pr+w2*nr)),w1,D),simplify(D,S).

% deriv(-log(1-(sigmoid(w3)*phgr+sigmoid(w4)*(1-phgr))),w3,D), simplify(D,S).
% deriv(-log(1-(sigmoid(w4) + (sigmoid(w3)-sigmoid(w4))*sigmoid(w0+w1*pr+w2*nr))),w1,D),simplify(D,S).

% deriv(-y*log(1-(1-p3)*(1-p1)^pr*(1-p2)^nr)-(1-y)*log((1-p3)*(1-p1)^pr*(1-p2)^nr),p3,D), simplify(D,S).
% deriv(-y*log(1-(1-p3)*(1-p1)^pr*(1-p2)^nr),p3,D), simplify(D,S).
% deriv(-(1-y)*log((1-p3)*(1-p1)^pr*(1-p2)^nr),p3,D), simplify(D,S).