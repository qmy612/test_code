clear all;
dscarry=zeros(1,25);
iuarry=zeros(1,25);
macc=0;miu=0;mdsc=0;
for i=1:25
    stri=num2str(i);
    
    il=strcat('label',stri,'.bmp');
    ir=strcat(stri,'kgrow2.bmp');
    la=imread(il);
    re=imread(ir);
    k=0;k1=0;k2=0;b=0;
for m=1:128
    for n=1:128
        if (la(m,n)==255)&&(re(m,n)==255)
            k=k+1;
        end
        if (la(m,n)==0)&&(re(m,n)==0)
            b=b+1;
        end
        if (la(m,n)==255)
            k1=k1+1;
        end
        if (re(m,n)==255)
            k2=k2+1;
        end
    end
end
n11=k;n00=b;n01=k2-k;n10=k1-k;
t1=k1;t0=128*128-k1;
dscarry(i)=(2*k)/(k1+k2);

iuarry(i)=0.5*(n11/(t1+n01)+n00/(t0+n10));

end
dsc=mean(dscarry);
iu=mean(iuarry);
