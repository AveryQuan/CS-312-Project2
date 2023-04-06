% CPSC 312 Calculus and Algebra in Prolog
% Copyright D. Poole 2023. Released under GPL https://www.gnu.org/licenses/gpl-3.0.en.html
  
% An expression can include algebaic variables, which are Prolog constants

%eval(Exp, Env, V) is true if expression Exp evaluates to V given environment Env
% An environment is a list of val(Var,Val) indicating that variable Var has value Val
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
    V is log(VA).
eval(exp(A),Env,V) :-
    eval(A,Env,VA),
    V is exp(VA),
    print("exp of").
eval(sigmoid(A),Env,V) :-
    eval(A,Env,VA),
    V is 1/(1+exp(-VA)).

% try:
% eval(aa*aa+b*11, [val(aa,3), val(b,7), val(dd,23)], V).
% eval(x+3*x+6*x*y+ 11*x*x, [val(x,7),val(y,-3),val(z,11)]).

%  Differentiation

% dv(E,X,DE) is true if DE is the derivative of E with respect to X
dv(X,X,1).
dv(C,X,0) :- atomic(C), dif(C,X).
dv(A+B,X,DA+DB) :- dv(A,X,DA), dv(B,X,DB).
dv(A-B,X,DA-DB) :- dv(A,X,DA), dv(B,X,DB).
dv(-A,X,-DA) :- dv(A,X,DA).
dv(A*B,X,DA*B+DB*A) :- dv(A,X,DA), dv(B,X,DB).
dv(exp(E),X,exp(E)*DE) :- dv(E,X,DE).
dv(F/G, X, (DF*G+DG*F)/G^2) :- dv(F,X,DF), dv(G,X,DG).
dv(sin(E),X,cos(E)*DE) :- dv(E,X,DE).

% smp(E,S) is true if S is a simplified version of expression E
smp(E,E) :- atomic(E).
smp(A+B,E) :- smp(A,AS), smp(B,BS), smps(AS+BS,E).
smp(A*B,E) :- smp(A,AS), smp(B,BS), smps(AS*BS,E).

% smps(E,S) is true if E is an expression with subparts simplified, S is it simplifies
smps(0+A,A).
smps(A+0,A).
smps(A+B,V) :- number(A), number(B), V is A+B.
smps(A+B,A+B).
smps(0*_,0).
smps(_*0,0).
smps(1*A,A).
smps(A*1,A).
smps(A*B,V) :- number(A), number(B), V is A*B.
smps(A*B,A*B).

% deriv(E,X,DE) is true if DE is the derivative of E with respect to X
deriv(X,X,1).
deriv(C,X,0) :- atomic(C), dif(C,X).
deriv(A+B,X,DA+DB) :-
    deriv(A,X,DA),
    deriv(B,X,DB).
deriv(A-B,X,DA-DB) :-
    deriv(A,X,DA),
    deriv(B,X,DB).
deriv(A*B,X,A*DB+B*DA) :-
    deriv(A,X,DA),
    deriv(B,X,DB).
deriv(A/B,X,(B*DA-A*DB)/(B*B)) :-
    deriv(A,X,DA),
    deriv(B,X,DB).

deriv(-A,X,-DA) :-
    deriv(A,X,DA).
deriv(A^B,X,B*(A^(B-1))*DA) :-  % only works when B does not involve X
    deriv(A,X,DA).
deriv(sin(E),X,cos(E)*DE) :-
    deriv(E,X,DE).
deriv(cos(E),X,-sin(E)*DE) :-
    deriv(E,X,DE).
deriv(exp(E),X,exp(E)*DE) :-
    deriv(E,X,DE).
deriv(log(E),X,DE/E) :-
  deriv(E,X,DE).
% sigmoid(X) = 1/(1+exp(-X))
deriv(sigmoid(E),X,sigmoid(E)*(1-sigmoid(E))*DE) :-
    deriv(E,X,DE).


% Some Examples to try:
%?- deriv(x+3*x+6*x*y, x, D).
%?- deriv(7+3*x+6*x*y, x, D).
%?- deriv(x+3*x+6*x*y+ 11*x*x, x, D).
%?- deriv(1/(1+exp(-x)),x,D), simplify(D,E).

% Multi-variate calculus:
%?- deriv(x+3*x+6*x*y+ 11*x*x, x, Dx), deriv(Dx,y,Dxy), simplify(Dxy,E).

% integ(E,X,IE) is true if IE is the integral of E with respect to X


% Basic Integral Rules
integ(0, X, 0).
integ(C, X, C*X) :- atomic(C), dif(C, X).
integ(X, X, X^2/2).
integ(X^N, X, (X^A)/A) :- atomic(N), A is N+1.
integ(1/X, X, ln(abs(X))).
integ((e/0)^X, X, (e/0)^X).
integ(C^X, X, (C^X)/ln(C)) :- atomic(C), dif(C,X).
integ(ln(X), X, X*ln(X)-X).
integ(A^X , X , A^X / ln(A) ):-
        atomic(A),
        dif(A, X).

% DEMO
% integ(0, X, 0).
% integ(5, X, 5*X).
% integ(X^4, X, (X^5)/5).

integ(C*A,X,C*IA) :-
    atomic(C),
    integ(A, X, IA).
integ(A+B,X,IA+IB) :-
    integ(A,X,IA),
    integ(B,X,IB).
integ(A-B,X,IA-IB) :-
    integ(A,X,IA),
    integ(B,X,IB).

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