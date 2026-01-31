from feature_generation import distance_and_angle_generator, hydrophobicity_generator, \
    physicochemical_feature_generator, ProtBERT_feature_generator, residue_coordinates_generator, \
    ProtXLNet_feature_generator, residue_accessibility_generator, label_generator, ESM_features_generator, \
    PSSM_generator, HMM_generator

from config import DefaultConfig
configs = DefaultConfig()
 
def generate_all_features():
    dataset_name = configs.dataset_name
    if dataset_name == 'dbd5':
        binding_type = 'u'
    else:
        binding_type = 'b'
    distance_and_angle_generator.generate_distance_and_angle_matrix(input_dir='./inputs/', dataset_name=dataset_name, protein_type='l', binding_type=binding_type)
    print('dist and angle l done')
    distance_and_angle_generator.generate_distance_and_angle_matrix(input_dir='./inputs/', dataset_name=dataset_name, protein_type='r', binding_type=binding_type)
    print('dist and angle r done')
    hydrophobicity_generator.generate_hydrophobicity(input_dir='./inputs/', dataset_name=dataset_name, protein_type='l', binding_type=binding_type)
    print('hydro l done')
    hydrophobicity_generator.generate_hydrophobicity(input_dir='./inputs/', dataset_name=dataset_name, protein_type='r', binding_type=binding_type)
    print('hydro r done')
    physicochemical_feature_generator.generate_physicochemical_features(input_dir='./inputs/', dataset_name=dataset_name, protein_type='l', binding_type=binding_type)
    print('phychem l done')
    physicochemical_feature_generator.generate_physicochemical_features(input_dir='./inputs/', dataset_name=dataset_name, protein_type='r', binding_type=binding_type)
    print('phychem r done')

    residue_coordinates_generator.generate_residue_coordinates(input_dir='./inputs/', dataset_name=dataset_name, protein_type='l', binding_type=binding_type)
    print('res acc l done')
    residue_coordinates_generator.generate_residue_coordinates(input_dir='./inputs/', dataset_name=dataset_name, protein_type='r', binding_type=binding_type)
    print('res acc r done')
    residue_accessibility_generator.generate_residue_accessibility(input_dir='./inputs/', dataset_name=dataset_name, protein_type='l', binding_type=binding_type)
    print('res acc l done')
    residue_accessibility_generator.generate_residue_accessibility(input_dir='./inputs/', dataset_name=dataset_name, protein_type='r', binding_type=binding_type)
    print('res acc r done')
    ESM_features_generator.generate_esm_features(input_dir='./inputs/', dataset_name=dataset_name, protein_type='l', binding_type=binding_type)
    print('ESM acc l done')
    ESM_features_generator.generate_esm_features(input_dir='./inputs/', dataset_name=dataset_name, protein_type='r', binding_type=binding_type)
    print('ESM acc r done')
    PSSM_generator.generate_pssm_features(input_dir='./inputs/', dataset_name=dataset_name, protein_type='l', binding_type=binding_type)
    print('PSSM acc l done')
    PSSM_generator.generate_pssm_features(input_dir='./inputs/', dataset_name=dataset_name, protein_type='r', binding_type=binding_type)
    print('PSSM acc r done')
    HMM_generator.generate_hmm_features(input_dir='./inputs/', dataset_name=dataset_name, protein_type='l', binding_type=binding_type)
    print('PSSM acc l done')
    HMM_generator.generate_hmm_features(input_dir='./inputs/', dataset_name=dataset_name, protein_type='r', binding_type=binding_type)
    print('PSSM acc r done')

    label_generator.generate_labels(input_dir='./inputs/', dataset_name=dataset_name, binding_type='b')
    print('label done')

if __name__ == '__main__':
    generate_all_features()
