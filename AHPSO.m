% -----------------------------------------------------------------------  %
% Altruistic Heterogeneous Particle Swarm Optimisation Algorithm                                     %
%
% Implemented by Fevzi Tugrul Varna - University of Sussex, 2021           %
% -------------------------------------------------------------------------%
% 
% Cite as: ----------------------------------------------------------------%
% F. T. Varna and P. Husbands, "AHPSO: Altruistic Heterogeneous Particle   %
% Swarm Optimisation Algorithm for Global Optimisation," 2021 IEEE         %
% Symposium Series on Computational Intelligence (SSCI), 2021, pp. 1-8,    %
% doi:10.1109/SSCI50451.2021.9660149.                                      %
% -----------------------------------------------------------------------  %
%% inputs: fhd,fId,n,d,range where fId=function no., n=swarm size, d=dimension, range=lower and upper bounds
%% e.g. AHPSO(fhd,5,60,30,[-100 100])
function [fmin] = AHPSO(fhd,fId,n,d,range)
PPM=false; %on/off state of the PPM mechanism
if PPM==true
    if d==10
        PPMpop=16;
    elseif d==30
        PPMpop=10;
    elseif d==50
        PPMpop=30;
    elseif d==100
        PPMpop=20;
    end
end

showProgress=true;
maxFES = 10^4*d;    % Maximum func evaluations
TMax = maxFES/n;    % Maximum Number of Iterations

Weight1 = 0.99 + (0.2-0.99)*(1./(1 + exp(-5*(2*(1:TMax)/TMax - 1)))); %Nonlinear decrease inertia weight(Sigmoid function)
C = 0.15;   %Modified constants of nonlinear decrease inertia weight
c1 = 2.5-(1:TMax)*2/TMax;  %personal acceleration coefficient
c2 = 0.5+(1:TMax)*2/TMax;  %social acceleration coefficient
alpha=20;  %number of potential lenders - set as 10-20 or regulate according to population size
ER=1;      %energy redistrubition rate
LB_rate=5; %lending/borrowing number reset rate
delta = 0; %number of agents in swarm with sufficient current energy level to activate
gamma=0.01;%controls the reshuffling period of paired particles

if PPM==true
    pairSize=2;
    M=reshape(randperm(PPMpop),PPMpop/pairSize,pairSize);
    M_alt = rand(1,PPMpop/2);  %initial alturistic values for paired particles
end

LB = range(1);
UB = range(2);

MaxVel = 0.15*(UB-LB);
MinVel = -MaxVel;

%% Initialisation
E_c = unifrnd(0.1,1,[1 n]);  %current energy level
E_a = unifrnd(0.5,1,[1 n]);  %activation energy level
L=randi(5,[1 n]);            %initial values for number of lents
B=randi(5,[1 n]);            %initial values number of borrows
PC=flip(1:n);                %prev couple (initial)
V=zeros(n,d);                %initial velocities
X=unifrnd(LB,UB,[n,d]);      %initial positions
PX=X;                        %initial pbest positions
F=feval(fhd,X',fId);         %function evaluation
PF=F;                        %initial pbest cost
GX=[];                       %gbest solution vector
GF=inf;                      %gbest cost

%update gbest
for i=1:n
    if PF(i)<GF, GF=PF(i); GX=PX(i,:); end
end

%% Main Loop of PSO
for t=1:TMax

    %alternatively, use beta=0.7 only
    if rand<0.5, beta=0.7; else, beta=mean(L./B); end

    %rearrange paired particles
    if PPM==true
        if mod(t,round(TMax*gamma))==0
            for ii=1:length(M)
                PC(M(ii,1))=M(ii,2);
                PC(M(ii,2))=M(ii,1);
            end
            M=reshape(randperm(PPMpop),PPMpop/pairSize,pairSize);
        end

        %calculate altruism values for each pair
        for jj=1:PPMpop/2
            M_alt(jj) = (L(M(jj,1))/B(M(jj,1))) + (L(M(jj,2))/B(M(jj,2)));
        end
    end

    delta=sum(E_c>=E_a); %update delta

    for i=1:n
        %update inertia weight
        if F(i) >= mean(F)
            w = Weight1(t) + C;
            if w>0.99,  w = 0.99;end
        else     % average_n < Average_g
            w = Weight1(t) - C;
            if w<0.20,  w = 0.20;end
        end

        if t<TMax*0.9
            %check if ith particle is able to activate
            if E_c(i) >= E_a(i)
                V(i,:) = w*V(i,:) + c1(t)*rand([1 d]).*(PX(i,:) - X(i,:)) + c2(t)*rand([1 d]).*(GX - X(i,:));
            else %particle needs to borrow energy
                %calculate probabilities
                phi=(L(i)/B(i))*(delta/n);    %calculate phi
                if phi<beta %doesn't qualify for energy sharing
                    V(i,:) = w*V(i,:) + c1(t)*rand([1 d]).*(PX(i,:) - X(i,:)) + c2(t)*rand([1 d]).*(PX(getAlturisticParticle(0,i,L,B),:) - X(i,:));
                else         %energy sharing takes place
                    lenders = randperm(n,alpha);%select potential lenders
                    lenders(find(lenders==i)) = []; %remove self from the lender (if included)
                    E_b = 0; %total borrowed energy from lenders
                    E_r = (E_a(i)-E_c(i))/alpha;    %required energy from each lender

                    %borrow energy from lenders
                    for j=1:length(lenders)
                        if E_c(lenders(j)) >= E_r                                  %if lender has energy to lent
                            E_c(lenders(j)) = E_c(lenders(j)) - E_r;      %deduct borrowed energy from the lender
                            E_b  = E_b + E_r;                                               %borroable energy from the jth particle
                            L(lenders(j)) = L(lenders(j)) + 1;            %update number of times the particle lent
                        end
                    end

                    E_c(i) = E_c(i) + E_b;                                %add borrowed energy
                    E_c(i) = min(E_c(i),1);                               %E_c not > 1
                    E_c(i) = max(E_c(i),0);                               %E_c not < 0
                    B(i) = B(i) + 1;                                      %update number of times particle borrowed

                    %check if energy level is sufficient after borrowing
                    if E_c(i) >= E_a(i)
                        V(i,:) = w*V(i,:) + c1(t)*rand([1 d]).*(PX(i,:) - X(i,:)) + c2(t)*rand([1 d]).*(PX(getAlturisticParticle(1,i,L,B),:) - X(i,:));
                    else
                        V(i,:) = w*V(i,:) + c1(t)*rand([1 d]).*(PX(i,:) - X(i,:)) + c2(t)*rand([1 d]).*(mean(X(lenders(1:round(alpha/2)),:)) - X(i,:)); % 
                        V(i,:) = V(i,:)*E_c(i);
                    end
                end
            end
        else  %final exploitation phase
            V(i,:) = w*V(i,:) + c1(t)*rand([1 d]).*(PX(i,:) - X(i,:)) + c2(t)*rand([1 d]).*(GX - X(i,:));
        end

        V(i,:) = max(V(i,:), MinVel); V(i,:) = min(V(i,:), MaxVel); % Apply Velocity Limits
        X(i,:) = X(i,:) + V(i,:);   % Update position
        X(i,:) = max(X(i,:), LB);  X(i,:) = min(X(i,:), UB);

        %paired particle model
        if PPM==true
            if t<TMax*0.9
                if isempty(find(M==i))==false %if particle is paired
                    [pId,sId] = find(M==i);       %get id of the ith particle and the paired particle
                    if sId==1, cId=2; else; cId=1; end
                    if randi(2)==1   %Coupling-based learning
                        if E_c(M(pId,sId)) >= E_a(M(pId,sId)) && E_c(M(pId,cId)) >= E_a(M(pId,cId))         %tightly coupled pair
                            V(i,:) = w*V(i,:) + c1(t)*rand([1 d]).*(X(M(pId,cId),:)*E_c(M(pId,sId)) - X(i,:)) + c2(t)*rand([1 d]).*(PX(M(pId,cId),:)*E_c(M(pId,sId)) - X(i,:));
                        elseif E_c(M(pId,sId)) < E_a(M(pId,sId)) && E_c(M(pId,cId)) < E_a(M(pId,cId))       %loosely coupled pair
                            V(i,:) = w*V(i,:) + c1(t)*rand([1 d]).*(PX(M(pId,sId),:)*E_c(M(pId,sId)) - X(i,:)) + c2(t)*rand([1 d]).*(X(M(pId,cId),:)*E_c(M(pId,sId)) - X(i,:));
                        elseif (E_c(M(pId,sId)) < E_a(M(pId,sId)) && E_c(M(pId,cId)) >= E_a(M(pId,cId))) || (E_c(M(pId,sId)) >= E_a(M(pId,sId)) && E_c(M(pId,cId)) < E_a(M(pId,cId)))   %neutral coupled pair
                            V(i,:) = w*V(i,:) + c1(t)*rand([1 d]).*(PX(M(pId,sId),:)*E_c(M(pId,sId)) - X(i,:)) + c2(t)*rand([1 d]).*(GX - X(i,:));
                        end
                    else %Opposition-based learning strategy
                        ith_A = L(M(pId,sId))/B(M(pId,sId)); %alturism value of the ith particle
                        jth_A = L(M(pId,cId))/B(M(pId,cId));

                        behaviour = randi(3);

                        if ith_A>jth_A
                            [~,alturist_pair_id]=max(M_alt);                                %id of the most alturistic pair
                            if behaviour==1
                                V(i,:) = w*V(i,:) + c1(t)*rand([1 d]).*(PX(i,:) - X(i,:)) + c2(t)*rand([1 d]).*(mean([PX(M(alturist_pair_id,1),:); PX(M(alturist_pair_id,2),:)]) - X(i,:));
                            elseif behaviour==2
                                alt_ids = [L(M(alturist_pair_id,1))/B(M(alturist_pair_id,1)) L(M(alturist_pair_id,2))/B(M(alturist_pair_id,2))];
                                [~,alturist_individual]=max(alt_ids);   %id of the most alturist individual of the most alturist pair
                                V(i,:) = w*V(i,:) + c1(t)*rand([1 d]).*(PX(i,:) - X(i,:)) + c2(t)*rand([1 d]).*(PX(M(alturist_pair_id,alturist_individual),:) - X(i,:));
                            elseif behaviour==3
                                alt_ids = [L(M(alturist_pair_id,1))/B(M(alturist_pair_id,1)) L(M(alturist_pair_id,2))/B(M(alturist_pair_id,2))];
                                [~,alturist_individual]=max(alt_ids);   %id of the most alturist individual of the most alturist pair
                                V(i,:) = w*V(i,:) + c1(t)*rand([1 d]).*(PX(i,:) - X(i,:)) + c2(t)*rand([1 d]).*(X(M(alturist_pair_id,alturist_individual),:) - X(i,:));
                            end
                        else
                            [~,alturist_pair_id]=min(M_alt); %id of the most alturistic pair
                            if behaviour==1
                                V(i,:) = w*V(i,:) + c1(t)*rand([1 d]).*(PX(i,:) - X(i,:)) + c2(t)*rand([1 d]).*(mean([PX(M(alturist_pair_id,1),:); PX(M(alturist_pair_id,2),:)]) - X(i,:));
                            elseif behaviour==2
                                alt_ids = [L(M(alturist_pair_id,1))/B(M(alturist_pair_id,1)) L(M(alturist_pair_id,2))/B(M(alturist_pair_id,2))];
                                [~,alturist_individual]=min(alt_ids);   %id of the most alturist individual of the most alturist pair
                                V(i,:) = w*V(i,:) + c1(t)*rand([1 d]).*(PX(i,:) - X(i,:)) + c2(t)*rand([1 d]).*(PX(M(alturist_pair_id,alturist_individual),:) - X(i,:));
                            elseif behaviour==3
                                alt_ids = [L(M(alturist_pair_id,1))/B(M(alturist_pair_id,1)) L(M(alturist_pair_id,2))/B(M(alturist_pair_id,2))];
                                [~,alturist_individual]=min(alt_ids);   %id of the most alturist individual of the most alturist pair
                                V(i,:) = w*V(i,:) + c1(t)*rand([1 d]).*(PX(i,:) - X(i,:)) + c2(t)*rand([1 d]).*(X(M(alturist_pair_id,alturist_individual),:) - X(i,:));
                            end
                        end

                        %check if pair needs to be abandoned
                        if (L(M(pId,cId))/B(M(pId,cId)))<mean(L./B)     %if paired particle is less alturist then avg
                            %randomly switch pair of the current particle
                            [prevPair,~]=find(M==M(pId,cId));
                            [newPair,~]=find(M==M(randi(PPMpop/pairSize),cId));
                            x_1 = M(pId,cId);
                            x_2 = M(newPair,cId);
                            M(prevPair,cId) = x_2;
                            M(newPair,cId) = x_1;
                        end
                    end
                    V(i,:) = max(V(i,:), MinVel); V(i,:) = min(V(i,:), MaxVel); % Apply Velocity Limits
                    X(i,:) = X(i,:) + V(i,:); % Update position
                    X(i,:) = max(X(i,:), LB); X(i,:) = min(X(i,:), UB); % Apply Lower and Upper Bound Limits
                end
            end
        end

        F(i) = feval(fhd,X(i,:)',fId); %function evalutation
        if F(i) < PF(i), PX(i,:) = X(i,:); PF(i) = F(i); end % Update Personal best
        if PF(i) < GF, GX = PX(i,:); GF = PF(i); end % Update Global best
    end

    %reset energy levels
    if mod(t,ER)==0
        E_c = unifrnd(0.1,1,[1 n]);                           %current energy level
        E_a = unifrnd(0.5,1,[1 n]);                           %activation energy level
    end

    %reinitialise lending/borrowing numbers
    if mod(t,LB_rate)==0
        L=randi(10,[1 n]);   %number of lents
        B=randi(10,[1 n]);   %number of borrows
    end

    % Display Iteration Information
    if showProgress
        disp(['Iteration ' num2str(t) ': best cost = ' num2str(GF)]);
    end

end
fmin = GF;
end

function [id] = getAlturisticParticle(request,i,L,B)
if request == 0
    L(i)=inf;             %remove ith particle from list
    B(i)=inf;
end

A_vals = L./B;            %altruism values for all particles
if request==0             %returns id of the least altruistic particle
    [~,id] = min(A_vals);
else                      %returns id of the most altruistic particle
    [~,id] = max(A_vals);
end
end