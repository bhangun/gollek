package tech.kayys.wayang.tool.repository;

import io.smallrye.mutiny.Uni;
import tech.kayys.wayang.tool.entity.AuthProfile;

import java.util.List;

public interface AuthProfileRepository {

    Uni<List<AuthProfile>> listAllProfiles();

    Uni<List<AuthProfile>> findByTenantId(String tenantId);

    Uni<List<AuthProfile>> findByTenantIdAndEnabled(String tenantId, boolean enabled);

    Uni<AuthProfile> findById(String profileId);

    Uni<AuthProfile> findByTenantIdAndProfileId(String tenantId, String profileId);

    Uni<AuthProfile> save(AuthProfile profile);

    Uni<AuthProfile> update(AuthProfile profile);

    Uni<Boolean> deleteById(String profileId);

    Uni<List<AuthProfile>> searchProfiles(String query, Object... params);

    Uni<Long> count();

    Uni<Long> countByTenantId(String tenantId);
}