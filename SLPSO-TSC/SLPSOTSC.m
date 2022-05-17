classdef SLPSOTSC < ALGORITHM
    % <multi> <real> <expensive>
    % surrogate-assistant evolutionary optimization by a decision classification
    % wmax --- 20 --- The maximum number of internal evluation
    
    %------------------------------- Copyright --------------------------------
    % Copyright (c) 2021 BIMK Group. You are free to use the PlatEMO for
    % research purposes. All publications which use this platform or any code
    % in the platform should acknowledge the use of "PlatEMO" and reference "Ye
    % Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
    % for evolutionary multi-objective optimization [educational forum], IEEE
    % Computational Intelligence Magazine, 2017, 12(4): 73-87".
    %--------------------------------------------------------------------------
    
    % This function is written by Qi-Te Yang, test for SAEA with one global
    % model and local model. The local model uses zoomed decision space.
    
    methods
        function main(Algorithm,Problem)
            
            %% Generate initial population based on Latin hypercube sampling
            InitN          = 10*Problem.D;
            P          = UniformPoint(InitN,Problem.D,'Latin');
            Population = SOLUTION(repmat(Problem.upper-Problem.lower,InitN,1).*P+repmat(Problem.lower,InitN,1));
            PopDec = Population.decs;
            PopObj = Population.objs;
            fmin = min(PopObj,[],1);
          
            %% model training
            % RBFN
            goal=sqrt(sum((max(PopObj)-min(PopObj)).^2))*0.005;
            net = newrb(PopDec', PopObj', goal, 1.0, size(PopDec,2), 1e+06);
            % Kriging model
            Model = cell(1,Problem.M);
            Theta = 5.*ones(Problem.M,Problem.D);
            for i = 1 : Problem.M
                [X_train,Y_train] = dsmerge(PopDec,PopObj(:,i));
                model = dacefit(X_train,Y_train,'regpoly0','corrgauss',Theta(i,:),1e-5.*ones(1,Problem.D),100.*ones(1,Problem.D));
                Theta(i,:) = model.theta;
                Model{i} = model;
            end
            
            %% Optimization
            Archive = Population;
            decs = Archive.decs;
            ArcDec = [];
            while size(ArcDec,1) < Problem.N
                offdec = OperatorGA(decs);
                ArcDec = [ArcDec;offdec];
            end

            while Algorithm.NotTerminated(Population)
                randIndex  = randperm(size(ArcDec,1));
                ArcDec = ArcDec(randIndex(1:Problem.N),:);
                %% Optimization
                % RBFN
                ArcObj = sim(net,ArcDec');
                ArcObj = ArcObj';
                w_point = 1.1.*max(ArcObj,[],1);
                Hv = Hypervolume_MEX(ArcObj, w_point);
                ArcObj_GP = zeros(Problem.N,Problem.M);
%                 ArcObj_PR = zeros(Problem.N,Problem.M);
                for i = 1 : Problem.N
                    for j = 1 : Problem.M
                        predX = ArcDec(i,:);
                        % Kriging
                        [ArcObj_GP(i,j),~,mse] = predictor(predX,Model{j});
                        snd = normpdf((fmin(j)-ArcObj_GP(i,j))/mse); % 正态概率密度函数 normal probability density function
                        SND = normcdf((fmin(j)-ArcObj_GP(i,j))/mse); % 正态累积分布函数 normal cumulative distribution function
                        ArcObj_GP(i,j) = (fmin(j)-ArcObj_GP(i,j))*SND + mse*snd; % EI
                    end
                end
                [~,Sort] = sort(ArcObj_GP,1,'descend');
                Gbest = [];
                for i = 1 : Problem.M
                    gbest = find(Sort(:,i)==1);
                    Gbest = [Gbest,gbest];
                end
                
                for i = 1 : Problem.N
                    Inddec = ArcDec(i,:);
                    Teadec = [];
                    Offdec = [];
                    for j = 1 : Problem.M
                        if Sort(i,j) > 1
                            teacher_sort = unidrnd(Sort(i,j));
                            teacher_index = find(Sort(:,j)==teacher_sort);
                            teadec = ArcDec(teacher_index,:);  
                            offdec = Inddec;
                            r1 = rand(Problem.D,1); r1 = r1'; 
                            r2 = rand(Problem.D,1); r2 = r2';
                            for kk = 1 : Problem.D
                                if rand(1) < 0.5
                                    r1(kk) = 0;
                                end
                                if rand(1) < 0.5
                                    r2(kk) = 0;
                                end
                            end
                            offdec = offdec + r1.*(teadec-offdec) + r2.*(ArcDec(Gbest(j),:)-offdec);
                            Offdec = [Offdec;offdec];
                            Teadec = [Teadec;teadec];
                        end
                    end

                    if ~isempty(Teadec)
                        offdec = Inddec;
                        arset = offdec;
                        for j = 1 : size(Teadec,1)
                            r1 = rand(Problem.D,1); r1 = r1';
                            r2 = rand(Problem.D,1); r2 = r2';
                            for kk = 1 : Problem.D
                                if rand(1) < 0.5
                                    r1(kk) = 0;
                                end
                                if rand(1) < 0.5
                                    r2(kk) = 0;
                                end
                            end
                            offdec = offdec + r1.*(Teadec(j,:)-arset) + r2.*(ArcDec(Gbest(j),:)-arset);
                        end
                        Offdec = [Offdec;offdec];
                    else
                        Offdec = Inddec;
                        id = unidrnd(Problem.D);
                        upper = Problem.upper;
                        lower = Problem.lower;
                        Offdec(id) = Offdec(id) + (upper(id)-lower(id))*normrnd(0,0.1.^2);
                    end
                    % 对每个目标offspring用RBFN prediction
                    Offobj = sim(net,Offdec');
                    Offobj = Offobj';
                    % 用HV贡献率判断哪个好
                    newpop = ArcObj;
                    contri_hv = zeros(1,size(Offobj,1));
                    for j = 1 : size(Offobj,1)
                        newpop(i,:) = Offobj(j,:);
                        contri_hv(j) = Hypervolume_MEX(newpop, w_point) - Hv;
                    end
                    if max(contri_hv) > 0
                        [~,change] = max(contri_hv);
                        ArcDec(i,:) = Offdec(change,:);
                    end
                end
                
                % termination
                Population = SOLUTION(ArcDec);
            end
        end
    end
end