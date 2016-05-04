
echo "Downloading..."

wget --no-check-certificate http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz
tar xvf tasks_1-20_v1-2.tar.gz
mv tasks_1-20_v1-2 bAbI_data
rm *.gz

echo "Done ."
