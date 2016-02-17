:-module(test_data, [print_data_files/2
		    ,reber//0
		    ,embedded_reber//0]).
:-use_module(library(clpfd)).

%!	validation_file(V) is det.
%
%	Name of the validation file to print out and read in (to hold
%	out a test sample).
data_file('test_data_reber.py').

%!	validation_data(V) is det.
%
%	Name of the list storing the validation data.
validation_data('v').

%!	training_sample(T) is det.
%
%	Name of the list storing the training samples.
training_sample('t').

%!	testing_sample(S) is det.
%
%	Name of the python list storing the testing samples.
testing_sample('s').

%!	reber_length(+L) is det.
%
%	Length of Reber grammar strings to generate.
reber_length(10).

%!	embedded_reber_length(+L) is det.
%
%	Length of embedded Reber grammar strings to generate.
embedded_reber_length(13).



%!	print_data_files(+Relation,+Training_sample) is det.
%
%	Generate training data and split into training and validation
%	sets, then print those sets out.
%
%	Training_sample is the fraction of the validation set size to
%	hold out for training. Relation is the relation to use to
%	generate the data.

%	Valid relations must generate lists of n+1 values, meant to be
%	n dependent variables and a response variable (but there's
%	nothing saying that the last variable must be the response or
%	that there has to be a single response; also, there may be a
%	single variable in which case you just have a time series).
%
%	Training and validation data is printed out as a python list, to
%	be read in immediately by a python program. The name of the
%	validation and training lists is taken from the predicates at
%	the start of this file.
%
print_data_files(Relation,Training_size):-
	(   Relation == reber
	->  reber_length(L)
	;   Relation  == embedded_reber
	->  embedded_reber_length(L)
	)
	,! % Commit if Relation is a Reber grammar, even if rest fails
	% For instance, if three is no string of length L in the grammar
	,length(Parms, L)
	,aggregate(set(Parms), (relation(Relation,Parms)), Ss)
	% This is actually a bit broken- it only gets a single element
	% of Ss. But that's OK- for testing our LSTM we only need one. Will
	% have to fix later.
	,aggregate(set(Bs)
		  ,(member(S, Ss)
		   ,reber_string_bit_vector(S, Bs))
		  ,Data)
	,print_to_data_file(Data, Training_size)
	,!. % No more results needed.

print_data_files(Relation,Training_size):-
	(   (Training_size > 1
	    ; Training_size < 0)
	->  throw('Float must be a number between 0 and 1')
	;   true
	)
	% Generate data from the given Relation
	,aggregate(set(Parms), (relation(Relation,Parms),label(Parms)), Data)
	,print_to_data_file(Data, Training_size)
	,!.


%!	print_to_data_file(+Data,+Training_size) is det.
%
%	Business end of print_data_files/2. Convenience predicate to
%	generalise parent (basically, to cover generating Reber grammar
%	files that are handled differently to numeric relations-
%	specifically because they don't need to be labelled by label/1).
print_to_data_file(Data, Training_size):-
% Count the number of validation examples:
	length(Data, Count)
	,Sample_size is floor(Count * Training_size)
	,sample(Data,Sample_size,Train_sample,Test_sample)
	,data_file(DF)
	,validation_data(V)
	,training_sample(T)
	,testing_sample(S)
	,open(DF,write,Stream,[])
	% Print the full data set as a python list
	,format(Stream,'~w = ~w~n',[V,Data])
	% Print the training sample as a python list
	,format(Stream, '~w = ~w~n', [T,Train_sample])
	% Print the testing sample as a python list
	,format(Stream,'~w = ~w~n', [S,Test_sample])
	,close(Stream)
	,!.

%!	sample(+Validation,+Train_size,+Train_sample,-Test_sample) is det.
%
%	Select a random Train_sample of size Train_size from list
%	Validation; Test_sample is the difference of Validation and
%	Train_sample.
%
%	TODO: Move to library
%
sample(Vs,T_size,T_sample,S_sample):-
	% Sort the list, to allow it to be treated as a set.
	% Relations should not return any duplicates (sorting will remove them).
	sort(Vs,Vs_)
	,unique_sample(Vs_,T_size,T)
	% Also make sure
	,sort(T,T_sample)
	,ord_subtract(Vs_,T_sample,S_sample).

%!	sample(+List,+Size,-Samples) is det.
%
%	Select a random Sample of a given Size from the elements of
%	List.
%
%	%TODO: Move me to libraries.
%
sample(Ls,S,Ss):-
	findall(L,
		(between(1,S,_)
		,random_member(L,Ls))
	       ,Ss).


%!	unique_sample(+List,+Size,-Samples) is det.
%
%	As sample/3 but selects samples without replacement.
%	Fails silently if Size is larger than the length of List.
%
%	TODO: Could add some unit tests, using all_distinct/1 from
%	library(clpfd).
%
%	TODO: Move to library
%
unique_sample(Ls,K,Ss):-
	length(Ls, N)
	% Make K random numbers up to the length of Ls
	,randset(K,N,Ns)
	,unique_samples(Ls,Ns,1,[],Ss).

%!	unique_samples(+List,+Indices,+Current_index,+Acc,-Bind) is det.
%
%	Business end of unique_sample/3. Indices is a list of randomly
%	selected indices of elements in List. Current_index should be
%	initialised to 1; it's incremented as the list is unwound.
%	Elements for which Current_index matches the current head of
%	Indices will be added to the Accumulator and bound to the
%	out-variable in the end.
%
unique_samples([],[],_,Ss,Ss_):-
	% Reorder the list by sort order
	reverse(Ss,Ss_)
	,!
	.
unique_samples([L|Ls],[N|Ns],N,Acc,Bind):-
	% If the index of L matches N, hold it out from List.
	N_ is N + 1
	,! % Don't backtrack into incrementing the index!
	,unique_samples(Ls, Ns, N_, [L|Acc], Bind).
unique_samples([_|Ls],Ns,N,Acc,Bind):-
	N_ is N + 1
	,!
	,unique_samples(Ls, Ns, N_, Acc, Bind).


%!	relation(+Relation,-Parameters) is det.
%
%	Enumerate the set of Parameters of a given Relation.
relation(unary,[X]):-
	 X in 1..100.

relation(binary,[X,Y]):-
	 N1 = 1..100
	,N2 = 1..100
	,X in N1
	,Y in N2.

relation(minimal,[X,Y,Z,A]):-
	 N1 = 1..3
	,N2 = 1..3
	,N3 = 1..3
	,X in N1
	,Y in N2
	,Z in N3
	,A in 1..2.

relation(polynomial,[X,Y,Z,A]):-
	 N1 = 1..10
	,N2 = 1..3
	,N3 = 1..6
	,X in N1
	,Y in N2
	,Z in N3
	,A #= X^2 + 2 * Y + Z.

relation(varied,[X,Y,Z,A]):-
	 N1 = 1..10
	,N2 = 1..3
	,N3 = 1..6
	,M = 1..4
	,X in N1
	,Y in N2
	,Z in N3
	,A in M.

relation(uniformish,[X,Y,Z,A]):-
	 N = 1..10
	,M = 1..4
        ,X in N
	,Y in N
	,Z in N
	,A in M.

relation(reber, Rs):-
	phrase(reber, Rs).

relation(embedded_reber, Rs):-
	phrase(embedded_reber, Rs).


%!	reber_string_bit_vector(+Reber_string,-Bit_vector) is det.
%
%	Translate between a Reber string (embedded or otherwise) and a
%	one-hot bit-vector representation of the same.
%
reber_string_bit_vector(Reber_string, Bit_vector):-
	reber_string_bit_vectors(Reber_string, [], Bit_vector).

%!	reber_string_bit_vectors(+String,+Acc,-Bind) is det.
%
%	Business end of reber_string_bit_vector/2. Delegates to
%	reber_bits//1 for translation and just takes care of
%	accumulating the bits.
%
reber_string_bit_vectors([], Bit_vector, Rotcev_tib):-
	reverse(Bit_vector, Rotcev_tib).
reber_string_bit_vectors([R|Rs], Acc, Bind):-
	phrase(reber_bits(R), Bs)
	,reber_string_bit_vectors(Rs, [Bs|Acc], Bind).


%!	reber_bits// is det.
%
%	Translation grammar from Reber characters to one-hot bit vector
%	representation. There are seven characters in the Reber language
%	therefore there are seven bits per vector.
%
reber_bits('B') --> [1,0,0,0,0,0,0].
reber_bits('T') --> [0,1,0,0,0,0,0].
reber_bits('S') --> [0,0,1,0,0,0,0].
reber_bits('X') --> [0,0,0,1,0,0,0].
reber_bits('P') --> [0,0,0,0,1,0,0].
reber_bits('V') --> [0,0,0,0,0,1,0].
reber_bits('E') --> [0,0,0,0,0,0,1].


%!	embedded_reber// is nondet.
%
%	Embedded Reber grammar.
%	Like the non-embedded Reber grammar but embedded.
%	Actually, the Reber grammar is embedded into this one, so
%	calling this one the "embedded" one is a bit of a misnomer. It's
%	really a grammar _embedding_ the Rebber grammar, though since a
%	guy called Reber invented this one too it's also, by definition,
%	a Reber grammar itself.
%
%	But it's not embedded. It's _embedding_. Mkay?
%
%	Everyone calls it "embedded" though so OK. It's used to test
%	that learners can learn long-term dependencies between tokens in
%	a string, so for example 'BT...PE' is not OK, as is 'BP...TE'.
%
embedded_reber --> embedded_rebber(b).

% Depth one
embedded_rebber(b) --> ['B'],embedded_rebber(t_1).
embedded_rebber(b) --> ['B'], embedded_rebber(p_1).
% Depths two and three
embedded_rebber(t_1) --> ['T'], reber, embedded_rebber(t_2).
embedded_rebber(p_1) --> ['P'], reber, embedded_rebber(p_2).
% Depth four
embedded_rebber(t_2) --> ['T'], embedded_rebber(e).
embedded_rebber(p_2) --> ['P'], embedded_rebber(e).
% Depth five
embedded_rebber(e) --> ['E'].

:-begin_tests(embedded_reber).

% This should fail- it's a naked non-embedded Reber string.
test(embedded_btssxxtvve, [fail]):-
	phrase(embedded_reber, ['B','T','S','S','X','X','T','V','V','E']).
% And this too (starting with BP)
test(embedded_bpvve, [fail]):-
	phrase(embedded_reber, ['B','P','V','V','E']).

% Example of a valid embedded reber string: starts with 'BT', continues
% with a non-embedded (though in truth, embedded) Reber string and ends
% with 'TE'
test(embedded_bt_btssxxtvve_te, [nondet]):-
	phrase(embedded_reber, ['B','T','B','T','S','S','X','X','T','V','V','E','T','E']).
% Similar but beginning with 'BP' and ending in 'PE'
test(embedded_bp_bpvve_pe, [nondet]):-
	phrase(embedded_reber, ['B','P','B','P','V','V','E','P','E']).


% Example of an invalid embedded reber string: starts with 'BT',
% continues with a Reber string but after all this effort ends with
% 'PE'. What a let-down.
test(embedded_bt_btssxxtvve_pe, [fail]):-
	phrase(embedded_reber, ['B','T','B','T','S','S','X','X','T','V','V','E','P','E']).
% Similar but beginning with 'BP' and ending in 'TE'
test(embedded_bp_bpvve_te, [fail]):-
	phrase(embedded_reber, ['B','P','B','P','V','V','E','T','E']).


:-end_tests(embedded_reber).

%!	reber// is nondet.
%
%	Not the embedded Reber grammar.
%	Rules can be read as representing a directed edge in a graph:
%	==
%	N(p) --> [L], N(n).
%	==
%
%	Where N(p) the last visited node, L a term annotating the
%	current arc and N(n) the node the arc is pointing to.
%
reber --> reber(b).

/* Depth one*/
reber(b) --> ['B'], reber(t).
reber(b) --> ['B'], reber(p_1).
% Depth two
reber(t) --> ['T'], reber(ss).
reber(t) --> ['T'], reber(x_1).
reber(p_1) --> ['P'], reber(ts).
reber(p_1) --> ['P'], reber(v_1).
% Depth three
reber(ss) --> ['S'], reber(x_1).
reber(ss) --> ['S'], reber(ss).
reber(ts) --> ['T'], reber(v_1).
reber(ts) --> ['T'], reber(ts).
% Depth four
reber(x_1) --> ['X'], reber(x_2).
reber(x_1) --> ['X'], reber(s).
reber(v_1) --> ['V'], reber(v_2).
reber(v_1) --> ['V'], reber(p_2).
% Depth five
reber(x_2) --> ['X'], reber(ts).
reber(x_2) --> ['X'], reber(v_1).
reber(s) --> ['S'], reber(e). % Also reachable at depth six
reber(v_2) --> ['V'], reber(e).
reber(p_2) --> ['P'], reber(s).
% Depth six
reber(p_2) --> ['P'], reber(x_2).
% Depth seven
reber(e) --> ['E'].

:-begin_tests(reber).
% Test strings taken from:
% https://www.willamette.edu/~gorr/classes/cs449/reber.html
% Probably not exactly comprehensive, but will do for now.

test(btssxxtvve, [nondet]):-
	phrase(reber, ['B','T','S','S','X','X','T','V','V','E']).
test(bpvve, [nondet]):-
	phrase(reber, ['B','P','V','V','E']).
test(btxxvpse, [nondet]):-
	phrase(reber, ['B','T','X','X','V','P','S','E']).
test(bpvpxvpxvpxvve, [nondet]):-
	phrase(reber, ['B','P','V','P','X','V','P','X','V','P','X','V','V','E']).
test(btsxxvpse, [nondet]):-
	phrase(reber, ['B','T','S','X','X','V','P','S','E']).

% Non-reber strings; tests must fail to pass.
test(btsspxse, [fail]):-
	phrase(reber, ['B','T','S','S','P','X','S','E']).
test(bptvvb, [fail]):-
	phrase(reber, ['B','P','T','V','V','B']).
test(btxxvvse, [fail]):-
	phrase(reber, ['B','T','X','X','V','V','S','E']).
test(bpvspse, [fail]):-
	phrase(reber, ['B','P','V','S','P','S','E']).
test(btssse, [fail]):-
	phrase(reber, ['B','T','S','S','S','E']).

:-end_tests(reber).








