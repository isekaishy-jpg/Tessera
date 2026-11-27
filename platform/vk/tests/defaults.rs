//! `Default` gives every tagged struct embedded by value in another struct its `sType`: the rows
//! vkgen generates (`generated/defaults.rs`, one per embedded field) and two cases written out,
//! one in an untagged parent and one in a tagged parent.

#[path = "generated/defaults.rs"]
mod defaults;

use tessera_vk::vk::{self, TaggedStructure};

#[test]
fn every_embedded_tagged_struct_gets_its_structure_type() {
    let rows = defaults::embedded_tags();
    assert!(!rows.is_empty(), "vkgen found no embedded tagged struct");
    for (field, actual, expected) in rows {
        assert_eq!(actual, expected, "{field}");
    }
}

#[test]
fn embedded_tags_in_untagged_and_tagged_parents() {
    let untagged = vk::SubpassSampleLocationsEXT::default();
    assert_eq!(
        untagged.sample_locations_info.s_type,
        vk::SampleLocationsInfoEXT::STRUCTURE_TYPE
    );
    let tagged = vk::ComputePipelineCreateInfo::default();
    assert_eq!(tagged.s_type, vk::ComputePipelineCreateInfo::STRUCTURE_TYPE);
    assert_eq!(
        tagged.stage.s_type,
        vk::PipelineShaderStageCreateInfo::STRUCTURE_TYPE
    );
}
