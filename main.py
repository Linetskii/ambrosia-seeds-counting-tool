from watershed import Sample

def main():
    # path = 'ambrosia2.png'
    # path = 'ambrosia.jpg'
    path='ambr_p_3.jpg'
    seeds = Sample(path)
    # seeds.show('image')
    seeds.show('gray')
    seeds.show('thresh')
    # seeds.slic()
    # seeds.morph_gac()
    # seeds.random_walker()
    # seeds.quickshift()
    seeds.watershed()
    # seeds.show('image')

if __name__ == '__main__':
    main()